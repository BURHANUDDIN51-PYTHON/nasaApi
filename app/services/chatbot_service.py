import google.generativeai as genai
import os
from app.config import settings

class ChatbotService:
    """
    A chatbot service that uses the Gemini API to provide concise answers.
    """

    # System prompt guiding the assistant to be concise, friendly, and handle small talk.
    SYSTEM_PROMPT = (
        "You are a friendly assistant that answers user questions concisely. "
        "Keep responses short and to the point unless the user asks for more detail. "
        "You may engage in light small talk and be empathetic, but always prioritize a clear, useful answer. "
        "When possible, use 1-3 short sentences or a short bullet list. "
        "Avoid long explanations and unnecessary background."
    )

    def __init__(self, api_key: str = None):
        """
        Initializes the ChatbotService.

        Args:
            api_key: Your Google API key for Gemini. If not provided, it will
                     try to get it from the GEMINI_API_KEY environment variable.
        """
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Fallback to environment variable if no API key is provided.
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
            except KeyError:
                raise ValueError(
                    "API key not found. Please provide it as an argument or set "
                    "the GEMINI_API_KEY environment variable."
                )
        # Initialize the Gemini Pro model and start a chat session with a system prompt
    
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        # Keep responses short by default (adjustable)
        self.max_output_tokens = 150
        # Start the chat with a system message so the assistant follows the concise, friendly style
        self.chat = self.model.start_chat(history=[
            {
                "role": "user",
                "parts": [{"text": self.SYSTEM_PROMPT}]
            },
            {
                "role": "model",
                "parts": [{"text": "Understood. I will provide a concise answer based on the context."}]
            }
        ])
    

    def get_concise_answer(self, query: str, context: str = "") -> str:
        """
        Gets a concise answer from the Gemini model based on a query and context.

        Args:
            query: The user's question.
            context: Optional context to provide to the model.

        Returns:
            A concise answer from the model.
        """
        if not query:
            return "Please provide a query."

        # Constructing a prompt for the model to generate a concise answer.
        prompt = (
            f"Based on the following context, please provide a concise answer for the user's query.\n\n"
            f"Context: {context}\n\n"
            f"User Query: {query}\n\n"
            f"Answer:"
        )

        try:
            # Ask the model with a low temperature for consistency and a token limit to keep answers concise.
            response = self.chat.send_message(prompt)
            # Some client wrappers return the generated text on `.text`, others on `.content` — prefer `.text`.
            text = getattr(response, "text", None) or getattr(response, "content", None) or str(response)
            return text.strip()
        except Exception as e:
            # Basic error handling for the API call
            return f"An error occurred: {e}"