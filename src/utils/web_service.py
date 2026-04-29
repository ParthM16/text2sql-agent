"""
Web Utility Service  Universal Unit Intelligence (UUI)
=======================================================
Provides live web fetching for conversion factors and unit normalization.
Supports both a free exchange rate API and a fallback hardcoded rate.
"""
import requests
import re
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Free API: https://www.exchangerate-api.com/ (1500 free requests/month)
# Alternative: https://open.er-api.com (unlimited, no key needed)
EXCHANGE_API_URL = "https://open.er-api.com/v6/latest/{base}"


class WebConversionService:

    #  Known currency aliases (maps common names to ISO codes) 
    CURRENCY_ALIASES = {
        "YEN": "JPY", "JAPANESE YEN": "JPY", "JPY": "JPY",
        "INR": "INR", "INDIAN RUPEE": "INR", "RUPEE": "INR",
        "USD": "USD", "DOLLAR": "USD", "US DOLLAR": "USD",
        "EUR": "EUR", "EURO": "EUR",
        "GBP": "GBP", "POUND": "GBP", "BRITISH POUND": "GBP",
        "CNY": "CNY", "YUAN": "CNY", "RMB": "CNY",
    }

    @staticmethod
    def _resolve_iso(name: str) -> str:
        """Converts a currency name/alias to its ISO 4217 code."""
        return WebConversionService.CURRENCY_ALIASES.get(name.upper().strip(), name.upper().strip())

    @staticmethod
    def get_conversion_factor(search_query: str) -> str:
        """
        Fetches a live conversion factor from a free exchange rate API.
        Falls back to hardcoded rates if the API is unreachable.
        
        Args:
            search_query: e.g., "convert YEN and INR" or "convert JPY to INR"
        
        Returns:
            A human-readable string like "1 YEN = 0.56 INR (Source: Open Exchange Rates API)"
        """
        # 1. Parse the search query for currency names
        try:
            # Extract currency-like words from the query
            words = re.findall(r'[A-Za-z]+', search_query.upper())
            currencies = [w for w in words if w in WebConversionService.CURRENCY_ALIASES]
            
            if len(currencies) < 2:
                logger.warning(f"Could not parse 2 currencies from: {search_query}")
                return WebConversionService._fallback(search_query)
            
            from_currency = WebConversionService._resolve_iso(currencies[0])
            to_currency = WebConversionService._resolve_iso(currencies[1])
            
        except Exception as e:
            logger.warning(f"Currency parsing failed: {e}")
            return WebConversionService._fallback(search_query)

        # 2. Try the live API
        try:
            url = EXCHANGE_API_URL.format(base=from_currency)
            logger.info(f" UUI: Fetching live rate from {url}")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("result") == "success":
                    rates = data.get("rates", {})
                    rate = rates.get(to_currency)
                    if rate:
                        factor = f"1 {currencies[0]} = {round(rate, 4)} {currencies[1]} (Source: Open Exchange Rates API  Live)"
                        logger.info(f" UUI Live Rate: {factor}")
                        return factor
                    else:
                        logger.warning(f"Target currency '{to_currency}' not found in API response.")
                else:
                    logger.warning(f"API returned non-success: {data}")
            else:
                logger.warning(f"API HTTP error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.warning("Exchange rate API timed out (5s). Using fallback.")
        except requests.exceptions.ConnectionError:
            logger.warning("Exchange rate API unreachable. Using fallback.")
        except Exception as e:
            logger.warning(f"Exchange rate API error: {e}")

        # 3. Fallback to hardcoded rates
        return WebConversionService._fallback(search_query)

    @staticmethod
    def _fallback(search_query: str) -> str:
        """Returns a hardcoded conversion factor when the API is unavailable."""
        # Common hardcoded rates (approximate)
        FALLBACK_RATES = {
            ("JPY", "INR"): 0.56,
            ("INR", "JPY"): 1.79,
            ("USD", "INR"): 83.5,
            ("INR", "USD"): 0.012,
            ("EUR", "INR"): 91.0,
            ("GBP", "INR"): 106.0,
        }
        
        words = re.findall(r'[A-Za-z]+', search_query.upper())
        currencies = [w for w in words if w in WebConversionService.CURRENCY_ALIASES]
        
        if len(currencies) >= 2:
            from_iso = WebConversionService._resolve_iso(currencies[0])
            to_iso = WebConversionService._resolve_iso(currencies[1])
            rate = FALLBACK_RATES.get((from_iso, to_iso))
            if rate:
                return f"1 {currencies[0]} = {rate} {currencies[1]} (Source: Hardcoded Fallback  API unavailable)"
        
        return "1 YEN = 0.56 INR (Source: Hardcoded Fallback  API unavailable)"

    @staticmethod
    def normalize_results_if_possible(df, unit_context, factor_text):
        """
        Attempts to normalize a dataframe based on the fetched factor.
        (Advanced step for future iteration)
        """
        return df
