"""LLM feedback helpers extracted from Untitled2.ipynb."""

from transformers import pipeline


class GPT2Coach:
    """Backward-compatible local coach wrapper using a Hugging Face model."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        # Local text2text model; no external API quota required.
        self.pipeline_llm = pipeline("text2text-generation", model=model_name)

    def generate_feedback(self, summary: str, max_new_tokens: int = 350) -> str:
        prompt = (
            "As an experienced badminton coach, analyze this player's performance "
            "data and provide constructive, actionable, and encouraging feedback. "
            "Focus on court coverage, movement efficiency, speed, and overall "
            "playing style. Consider perspective adjustment.\n\n"
            f"Performance Summary:\n{summary}\n\nCoaching Feedback:\n"
        )

        output = self.pipeline_llm(
            prompt,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
        return output[0]["generated_text"].strip()


def compare_players(player1_features: str, player2_features: str, analyzer) -> str:
    """Compare two players and return coaching guidance."""
    prompt = (
        "Compare the following two players' technical performance in detail. "
        "Analyze movement, angles, and technique. Give actionable, advanced "
        "coaching feedback for improvement.\n\n"
        f"Player 1 Data: {player1_features}\n"
        f"Player 2 Data: {player2_features}\n\n"
        "Output: Provide a professional comparison and improvement advice."
    )
    feedback = analyzer(prompt)
    if prompt.strip() in feedback.strip():
        return feedback.replace(prompt.strip(), "").strip()
    return feedback
