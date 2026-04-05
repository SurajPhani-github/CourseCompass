import json
import os
import requests
import logging

log = logging.getLogger(__name__)

# Fallback structure if API fails
FALLBACK_NOTES = {
    "topic_summary": "This is a fallback summary because the LLM API is unavailable.",
    "key_concepts": ["Concept 1: Definition", "Concept 2: Definition"],
    "revision_notes": "Important formulas and gotchas go here.",
    "interview_cheat_sheet": "1. What is X? -> Y. 2. Compare X and Z.",
    "self_check_questions": ["What is the main use case?", "How do you optimize it?"]
}

def generate_notes(topic: str, difficulty: str = "Beginner", goal: str = "Academic", context: str = "") -> dict:
    """
    Generates structured study notes using Llama-3 via Groq.
    Returns a dictionary of structured note formats.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.warning("No GROQ_API_KEY found, returning fallback notes.")
        return FALLBACK_NOTES

    prompt = f"""You are an expert AI tutor. 
Generate comprehensive, highly structured study notes for the topic '{topic}'.
Target Audience Difficulty: {difficulty}
Target Goal: {goal}
Context snippet: {context}

Return the response PURELY as a valid JSON object matching this schema:
{{
    "topic_summary": "A 2-paragraph overview",
    "key_concepts": ["List of core terms and definitions"],
    "revision_notes": "Detailed bullet points covering the mechanics and architecture",
    "interview_cheat_sheet": "Top 3 rapid-fire questions and answers for job interviews",
    "self_check_questions": ["List of 3 questions to test knowledge"]
}}

Do NOT include markdown block formatting (like ```json). Just the raw JSON object.
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a JSON-only response engine."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"].strip()
        
        # Clean potential markdown
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("\n", 1)[0]
            
        return json.loads(raw)
    except Exception as e:
        log.warning(f"Error calling Groq for notes: {e}")
        return FALLBACK_NOTES
