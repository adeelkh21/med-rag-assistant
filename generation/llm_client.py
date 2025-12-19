"""
STEP 7: Groq API Client

Groq LLaMA-3 70B integration for deterministic answer generation.

Model: llama3-70b-8192
Parameters: temperature=0.1, max_tokens=600, top_p=1.0
Behavior: Deterministic, no streaming
"""

from typing import Optional
import os
from pathlib import Path
from groq import Groq

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


# Model configuration (STRICT)
# Updated Dec 2024: Using llama-3.3-70b-versatile (production model)
MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.1  # Low for determinism
DEFAULT_MAX_TOKENS = 600
DEFAULT_TOP_P = 1.0


class GroqClient:
    """
    Client for Groq LLaMA-3 70B API.
    
    Configured for deterministic medical answer generation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (or uses GROQ_API_KEY env var)
        """
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = Groq(api_key=api_key)
        self.model = MODEL_NAME
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P
    ) -> str:
        """
        Generate answer using Groq LLaMA-3 70B.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query with context
            temperature: Sampling temperature (0.0-2.0, lower=more deterministic)
            max_tokens: Maximum response length
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated answer text (no metadata)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False  # No streaming (per spec)
            )
            
            # Extract only the generated text
            answer = response.choices[0].message.content
            return answer.strip()
        
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {e}")
    
    def get_model_name(self) -> str:
        """Get the model name being used."""
        return self.model


def create_client(api_key: Optional[str] = None) -> GroqClient:
    """
    Convenience function to create a Groq client.
    
    Args:
        api_key: Optional API key (uses env var if not provided)
    
    Returns:
        GroqClient instance
    """
    return GroqClient(api_key=api_key)


# Example usage
if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Groq Client Test")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n✗ GROQ_API_KEY environment variable not set")
        print("  Set it with: export GROQ_API_KEY='your-key-here'")
        sys.exit(1)
    
    print(f"\n✓ API key found")
    
    # Create client
    try:
        client = create_client()
        print(f"✓ Client initialized")
        print(f"  Model: {client.get_model_name()}")
    except Exception as e:
        print(f"\n✗ Failed to initialize client: {e}")
        sys.exit(1)
    
    # Test with simple prompt
    print("\nTesting API call...")
    system = "You are a helpful assistant."
    user = "Say 'Hello, World!' and nothing else."
    
    try:
        response = client.generate(system, user, temperature=0.0, max_tokens=50)
        print(f"✓ API call successful")
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"\n✗ API call failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ Groq client ready for use")
    print("=" * 70)
