"""
Multi-Model LLM Router
======================
Unified interface for routing inference across cloud and local LLMs:
  - Google Gemini 2.5 Flash (via google-genai Free Tier or Vertex AI)
  - Ollama Local (e.g. qwen2.5-coder:3b, llama3.2:3b via http://localhost:11434)
  - Simulated commercial cost telemetry for GPT-4o-mini and Claude 3.5 Haiku
  - Automatic graceful fallback to Gemini if local Ollama daemon is unreachable
"""
import os
import time
from typing import Optional, Dict, Any, Tuple
from src.agent.exceptions import LLMError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard Commercial Pricing Reference (USD per 1,000,000 tokens)
PRICING_TABLE = {
    "gemini-2.5-flash": {"input_per_m": 0.075, "output_per_m": 0.30},
    "gpt-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60},
    "claude-3-5-haiku": {"input_per_m": 0.80, "output_per_m": 4.00},
    "ollama-local": {"input_per_m": 0.0, "output_per_m": 0.0},
}


class LLMRouter:
    """Routes LLM inference requests to configured cloud or local providers."""

    def __init__(self, default_provider: Optional[str] = None):
        self.default_provider = default_provider or os.getenv("LLM_PROVIDER", "gemini-2.5-flash").strip()
        self._gemini_client = None
        self._init_gemini_client()

    def _init_gemini_client(self) -> None:
        """Initializes the google-genai client for Gemini/Vertex AI mode."""
        try:
            from google import genai
            project_id = os.getenv("PROJECT_ID", "").strip()
            location = os.getenv("LOCATION", "us-central1").strip()
            
            gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
            google_key = os.getenv("GOOGLE_API_KEY", "").strip()
            api_key = gemini_key or google_key
            if api_key == "your_api_key_here":
                api_key = ""

            # Prevent google-genai SDK warning:
            # "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY."
            # and ensure duplicate env vars in memory do not cause stale key lookups:
            if gemini_key and google_key:
                os.environ.pop("GOOGLE_API_KEY", None)
                os.environ["GEMINI_API_KEY"] = api_key
            elif api_key:
                os.environ["GEMINI_API_KEY"] = api_key
                os.environ.pop("GOOGLE_API_KEY", None)

            if project_id:
                try:
                    self._gemini_client = genai.Client(
                        vertexai=True,
                        project=project_id,
                        location=location,
                    )
                except Exception as e:
                    if api_key:
                        self._gemini_client = genai.Client(api_key=api_key)
                    else:
                        raise e
            elif api_key:
                self._gemini_client = genai.Client(api_key=api_key)
            else:
                self._gemini_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}")
            self._gemini_client = None

    def _calculate_costs(self, input_tokens: int, output_tokens: int, provider: str) -> Tuple[float, Dict[str, float]]:
        """Calculates actual cost and simulated comparison across commercial models."""
        comparisons = {}
        for model_name, rates in PRICING_TABLE.items():
            in_cost = (input_tokens / 1_000_000.0) * rates["input_per_m"]
            out_cost = (output_tokens / 1_000_000.0) * rates["output_per_m"]
            comparisons[model_name] = round(in_cost + out_cost, 6)

        # Actual estimated cost based on active provider
        if "ollama" in provider.lower():
            actual_cost = 0.0
        elif "gemini" in provider.lower():
            # Free tier effective cost is $0.00
            actual_cost = 0.0
        else:
            actual_cost = comparisons.get(provider, 0.0)

        return actual_cost, comparisons

    def _generate_gemini(self, prompt: str, system_instruction: Optional[str] = None) -> Tuple[str, int, int]:
        """Executes generation against Google Gemini."""
        if not self._gemini_client:
            self._init_gemini_client()
        if not self._gemini_client:
            raise LLMError("Gemini client not configured. Set GOOGLE_API_KEY or PROJECT_ID in .env")

        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

        try:
            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )
        except Exception as e:
            # If Vertex AI failed due to billing/permissions, attempt fallback to Gemini API Key if set
            api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
            if api_key and api_key != "your_api_key_here":
                try:
                    from google import genai
                    fallback_client = genai.Client(api_key=api_key)
                    response = fallback_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=full_prompt,
                    )
                    self._gemini_client = fallback_client
                except Exception:
                    raise e
            else:
                raise e

        text = getattr(response, "text", None)
        if text is None:
            candidates = getattr(response, "candidates", None)
            if candidates and len(candidates) > 0:
                text = getattr(candidates[0], "content", "")
        text = text.strip() if text else ""

        # Extract tokens
        input_tokens = 0
        output_tokens = 0
        um = getattr(response, "usage_metadata", None)
        if um:
            input_tokens = getattr(um, "prompt_token_count", 0) or 0
            output_tokens = getattr(um, "candidates_token_count", 0) or 0

        # Heuristic fallback if usage_metadata was omitted
        if input_tokens == 0:
            input_tokens = max(1, len(full_prompt) // 4)
        if output_tokens == 0 and text:
            output_tokens = max(1, len(text) // 4)

        return text, input_tokens, output_tokens

    def _generate_ollama(self, model_name: str, prompt: str, system_instruction: Optional[str] = None) -> Tuple[str, int, int]:
        """Executes generation against local Ollama instance."""
        import ollama

        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

        res = ollama.generate(
            model=model_name,
            prompt=full_prompt,
        )

        text = getattr(res, "response", "") or ""
        text = text.strip()

        input_tokens = getattr(res, "prompt_eval_count", 0) or 0
        output_tokens = getattr(res, "eval_count", 0) or 0

        if input_tokens == 0:
            input_tokens = max(1, len(full_prompt) // 4)
        if output_tokens == 0 and text:
            output_tokens = max(1, len(text) // 4)

        return text, input_tokens, output_tokens

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Unified inference entry point across all supported models.
        
        Args:
            prompt: User/task prompt.
            system_instruction: Optional system instruction/context.
            model_provider: Provider string (e.g. "gemini-2.5-flash" or "ollama/qwen2.5-coder:3b").
                            If None, uses default_provider.
                            
        Returns:
            Dictionary matching the standardized upgrade specification.
        """
        provider = (model_provider or self.default_provider or "gemini-2.5-flash").strip()
        start_time = time.perf_counter()
        fallback_used = False
        fallback_message = None

        text = ""
        input_tokens = 0
        output_tokens = 0

        # ── Ollama Local Route ───────────────────────────────────
        if "ollama" in provider.lower():
            if "/" in provider:
                model_name = provider.split("/", 1)[1]
            elif ":" in provider and not provider.startswith("http"):
                parts = provider.split(":", 1)
                model_name = parts[1] if parts[0].lower() == "ollama" else provider
            else:
                model_name = "qwen2.5-coder:3b"

            try:
                text, input_tokens, output_tokens = self._generate_ollama(
                    model_name=model_name,
                    prompt=prompt,
                    system_instruction=system_instruction
                )
            except Exception as e:
                # 🧠 Graceful Fallback: Switch to Gemini if Ollama is unreachable
                logger.warning(f"Ollama local model '{model_name}' unreachable ({e}). Safely falling back to Gemini 2.5 Flash...")
                fallback_used = True
                fallback_message = f"Local Ollama ({model_name}) was unreachable ({e}). Safely fell back to Gemini 2.5 Flash."
                provider = "gemini-2.5-flash"
                try:
                    text, input_tokens, output_tokens = self._generate_gemini(
                        prompt=prompt,
                        system_instruction=system_instruction
                    )
                except Exception as gemini_err:
                    raise LLMError(
                        "Both Ollama and Gemini fallback failed.",
                        internal_detail=f"Ollama err: {e}; Gemini err: {gemini_err}"
                    )

        # ── Gemini Cloud Route ───────────────────────────────────
        else:
            try:
                text, input_tokens, output_tokens = self._generate_gemini(
                    prompt=prompt,
                    system_instruction=system_instruction
                )
            except Exception as e:
                # 🧠 Graceful Fallback: Switch to local Ollama if Gemini fails
                err_detail = str(e)
                print(f"[FALLBACK] Gemini API call failed: {err_detail}. Gracefully falling back to local Ollama (qwen2.5-coder:3b)...", flush=True)
                logger.warning(f"Gemini API call failed ({err_detail}). Gracefully falling back to local Ollama (qwen2.5-coder:3b)...")
                fallback_used = True
                fallback_message = f"Gemini API unavailable ({err_detail}). Gracefully fell back to local Ollama (qwen2.5-coder:3b)."
                provider = "ollama/qwen2.5-coder:3b"
                try:
                    text, input_tokens, output_tokens = self._generate_ollama(
                        model_name="qwen2.5-coder:3b",
                        prompt=prompt,
                        system_instruction=system_instruction
                    )
                except Exception as ollama_err:
                    logger.exception("Both Gemini and local Ollama fallback failed")
                    raise LLMError(
                        "Both Gemini and local Ollama fallback failed. Please check your AI configuration.",
                        internal_detail=f"Gemini err: {e}; Ollama err: {ollama_err}"
                    )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        total_tokens = input_tokens + output_tokens
        actual_cost, cost_comparisons = self._calculate_costs(input_tokens, output_tokens, provider)

        return {
            "text": text,
            "provider": provider,
            "latency_ms": round(elapsed_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            # Backward-compatibility for workflow.py tokens_agg
            "prompt_tokens": input_tokens,
            "response_tokens": output_tokens,
            "estimated_cost_usd": actual_cost,
            "cost_comparison_usd": {
                "gemini-2.5-flash (free tier)": 0.0,
                "gemini-2.5-flash (nominal)": cost_comparisons.get("gemini-2.5-flash", 0.0),
                "ollama-local": 0.0,
                "gpt-4o-mini (est)": cost_comparisons.get("gpt-4o-mini", 0.0),
                "claude-3-5-haiku (est)": cost_comparisons.get("claude-3-5-haiku", 0.0),
            },
            "fallback_used": fallback_used,
            "fallback_message": fallback_message
        }
