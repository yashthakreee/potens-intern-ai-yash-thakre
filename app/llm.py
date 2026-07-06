"""
Thin wrapper around whichever LLM provider is configured (Groq or Gemini).
Both have a free tier, per the assignment brief. Swapping providers is just
an env var change (GEN_PROVIDER) -- nothing else in the app needs to know
which one is active.
"""
from app import config


def _call_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    resp = model.generate_content(prompt)
    return resp.text


def llm_available() -> bool:
    if config.GEN_PROVIDER == "groq":
        return bool(config.GROQ_API_KEY)
    return bool(config.GEMINI_API_KEY)


def generate(prompt: str) -> str:
    if not llm_available():
        raise RuntimeError(
            "No LLM API key configured. Copy .env.example to .env and set "
            "GROQ_API_KEY (or GEMINI_API_KEY with GEN_PROVIDER=gemini)."
        )
    if config.GEN_PROVIDER == "groq":
        return _call_groq(prompt)
    return _call_gemini(prompt)
