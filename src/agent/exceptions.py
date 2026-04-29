class AgentError(Exception):
    """Base exception  carries a user-safe message + internal detail."""
    def __init__(self, user_message, internal_detail=""):
        self.user_message = user_message       # shown in Streamlit UI
        self.internal_detail = internal_detail  # logged to file only
        super().__init__(user_message)

class LLMError(AgentError):
    """Raised when the LLM API fails or times out."""
    pass

class DatabaseError(AgentError):
    """Raised when DB connection or SQL execution fails."""
    pass

class ValidationError(AgentError):
    """Raised for bad user input (empty, too long, etc)."""
    pass

class SecurityError(AgentError):
    """Raised if generated SQL contains blocked operations (DROP, INSERT, etc)."""
    pass

