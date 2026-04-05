import json
import logging
from groq import Groq
from src.recommender.llm_branching import GROQ_API_KEY, CHAT_MODEL

log = logging.getLogger(__name__)

class AssessmentEngine:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = CHAT_MODEL
        
    def generate_quiz(self, topic: str, difficulty: str) -> dict:
        """
        Dynamically generates a 3-question MCQ quiz plus 1 scenario-based question
        using the Groq Llama-3 model based on user-selected difficulty.
        """
        prompt = f"""You are an expert technical examiner. Create an assessment for the following topic:
Topic: {topic}
Difficulty Selected: {difficulty} (Constraints: Easy=Basic definitions, Medium=Implementation logic, Difficult=Advanced architecture/Edge cases)

Generate a JSON object strictly following this schema:
{{
    "mcq_questions": [
        {{
            "question": "string",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "Exact string matching one option",
            "explanation": "Why this is correct"
        }}
    ],
    "scenario_question": {{
        "scenario": "string paragraph",
        "task": "string",
        "evaluation_rubric": "What to look for in a good answer"
    }}
}}

Provide exactly 3 MCQs. Return ONLY raw JSON. No markdown backticks."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You output only structured JSON."},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048,
            )
            text = response.choices[0].message.content.strip()

            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            return json.loads(text.strip())
        except Exception as e:
            log.error(f"Assessment generation failed: {e}")
            return {"error": str(e)}
