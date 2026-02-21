import google.generativeai as genai
import os

def generate_coaching_feedback(player_analysis_summary: str, api_key: str, model_name: str = 'gemini-pro'):
    """
    Generates coaching feedback using a Google Generative AI model based on player performance summary.

    Args:
        player_analysis_summary (str): A detailed summary of the player's performance.
        api_key (str): Your Google Generative AI API Key.
        model_name (str): The name of the Gemini model to use (default is 'gemini-pro').

    Returns:
        str: The generated coaching feedback.
    """
    if not api_key:
        return "Error: Google Generative AI API Key not provided."

    genai.configure(api_key=api_key)

    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return f"Error loading Gemini model '{model_name}': {e}"

    llm_prompt = f"""As an experienced badminton coach or analyst, analyze the provided player performance data and generate constructive, actionable, and encouraging feedback on the primary player's performance.
Your feedback should cover the player's court coverage, movement efficiency, speed, and overall playing style.
Please specifically consider how the perspective adjustment applied during the analysis makes the movement metrics more realistic. Use the following summary for your analysis:

{player_analysis_summary}

Based on this, provide your coaching feedback."""

    print("\n--- Generating Coaching Feedback using Google Generative AI ---")
    try:
        response = model.generate_content(llm_prompt)
        feedback = response.text
        print("Feedback generated successfully.")
        return feedback
    except Exception as e:
        return f"Error generating feedback: {e}"
