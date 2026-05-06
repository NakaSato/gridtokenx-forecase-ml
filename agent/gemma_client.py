import os
from typing import List, Dict, Any, Optional

# Attempt to load google-genai for Gemini proxy.
# In a real environment, this might connect to a local llama.cpp or HuggingFace server.
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class GemmaClient:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        if HAS_GENAI and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates text from the LLM."""
        if not self.client:
            return self._mock_generate(prompt)
            
        try:
            config = types.GenerateContentConfig(
                temperature=0.2,
            )
            if system_instruction:
                config.system_instruction = system_instruction
                
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            print(f"LLM API Error: {e}")
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        """Fallback mock generator for prototype demonstration without API keys."""
        if "incident" in prompt.lower() or "action plan" in prompt.lower():
            return (
                "🚨 **EARLY WARNING ACTION PLAN**\n\n"
                "1. [Immediate] Pre-charge BESS using excess mainland power. (Ref: SOP-BESS-2569-12 Section 4.1)\n"
                "2. [+15 min] Begin warm-up sequence for Diesel Generator DG-1. (Ref: SOP-DG-2569-04 Section 3.2)\n"
                "3. [+30 min] Commit DG-1 to cover load deficit if PV drops below 2 MW.\n\n"
                "If not adopted, expect a 5-minute brownout pulse during the peak transition."
            )
        elif "schedule" in prompt.lower() or "diff" in prompt.lower():
            return (
                "**Counterfactual Comparison:**\n\n"
                "By adopting the MILP recommended schedule, you save **145 liters/day** compared to manual operation.\n"
                "- 14:00-16:00: Used BESS instead of Diesel.\n"
                "- 19:00-21:00: Staged DG units more efficiently, reducing ramp loss.\n"
                "**Yearly ROI:** ~52,800 liters saved, amounting to approximately 1.85M THB/year."
            )
        return "I am the Gemma 4 Edge Agent. How can I assist you with the GridTokenX operations today?"

# Singleton instance
gemma_client = GemmaClient()
