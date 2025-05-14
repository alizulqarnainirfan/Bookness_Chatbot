import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel(model_name="gemini-2.0-flash")


def llm_response_stream(prompt: str):
    
    """
    Streams the LLM output response chunk-by-chunk for real-time user experience.
    """
    stream = llm.generate_content(prompt, stream=True)
    
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def llm_response(prompt: str) -> str:
    
    """
    Get a complete LLM response (non-streamed).
    """
    response = llm.generate_content(prompt)
   
    return response.text.strip()
