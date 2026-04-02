import json
import logging
from groq import Groq

log = logging.getLogger(__name__)

# Groq API - Free tier, no billing required
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Model choices - both free on Groq
CHAT_MODEL = "llama-3.3-70b-versatile"   # Fast, high-quality chat
BRANCH_MODEL = "llama-3.3-70b-versatile"  # Same model for roadmap generation


def generate_roadmap_branches(goal: str, learner_profile: dict = None) -> list[dict]:
    """
    Calls Groq (Llama 3.3 70B) to generate 3 distinct roadmap paths for the given goal.
    Returns a list of exactly 3 dictionaries.
    """
    profile_str = json.dumps(learner_profile) if learner_profile else "No existing profile."

    prompt = f"""You are an expert AI curriculum architect. A learner wants to master: '{goal}'.
Learner profile: {profile_str}

Construct exactly 3 highly distinct educational approaches:
1. Research / Theory-Heavy Path - deep math, theoretical papers, foundational principles
2. Applied / Engineering Path - infrastructure, APIs, frameworks, robust practices
3. Fast-Track / Builder Path - speed-to-MVP, rapid deployment, hacking apps together

Output STRICTLY as a valid JSON array of exactly 3 objects. Each object must follow this schema:
{{
    "path_name": "string",
    "description": "string",
    "target_projects": ["string", "string"],
    "stages": [
        {{
            "level": "Foundation|Core|Intermediate|Advanced|Specialization",
            "title": "string",
            "description": "string",
            "milestone": "string",
            "outcome_tags": ["string", "string", "string"]
        }}
    ]
}}

Return ONLY the raw JSON array. No markdown. No explanation. Minimum 4 stages per path."""

    try:
        response = client.chat.completions.create(
            model=BRANCH_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096,
        )
        text = response.choices[0].message.content.strip()

        # Strip markdown fences if model included them
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return json.loads(text.strip())
    except Exception as e:
        log.error(f"Groq API error during branch generation: {e}")
        return []


def get_chatbot_response(user_message: str, chat_history: list, context_str: str = "") -> str:
    """
    Calls Groq (Llama 3.3 70B) to respond to student questions.
    chat_history: list of dicts with keys 'role' ('user'|'assistant') and 'content'.
    context_str: optional extra roadmap context to inject as system knowledge.
    """
    try:
        system_content = """You are CourseCompass AI, an expert learning path advisor and curriculum designer.
You help students navigate their personalized learning roadmaps, suggest study strategies, recommend projects, and explain complex topics in an encouraging, concise way.
You have deep knowledge of AI, machine learning, data science, software engineering, and computer science curricula.
When a roadmap context is provided, use it to give highly specific, stage-by-stage advice.
Keep responses concise (3-5 sentences), actionable, and motivating."""

        # If there's explicit context (from Attach button), prepend it as system note
        if context_str and not context_str.startswith("[SYSTEM"):
            system_content += f"\n\nCurrent Learner Roadmap Context:\n{context_str}"

        messages = [{"role": "system", "content": system_content}]

        # Inject roadmap context from chat history (from the Attach button)
        for msg in chat_history:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            # Skip the raw [SYSTEM INJECTED] message in display, but include its content in the system prompt
            if content.startswith("[SYSTEM INJECTED Roadmap Context]"):
                # Add roadmap info into the system prompt instead
                messages[0]["content"] += f"\n\nAttached Roadmap from User:\n{content}"
                continue

            # Map "assistant" to correct role
            openai_role = "assistant" if role == "assistant" else "user"
            messages.append({"role": openai_role, "content": content})

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        log.error(f"Groq Chatbot API error: {e}")
        return f"Sorry, I encountered an error connecting to the AI: {e}"
