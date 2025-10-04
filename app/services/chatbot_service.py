import google.generativeai as genai
import os
from app.config import settings
from app.services.rag_service import RAGService

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
        
        self.ragServe = RAGService()
    

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
        
        print(f"Retrieving context for query: '{query}'...")
        retrieved_texts, source_papers = self.ragServe.query(query, top_k=100)
            
            
        if not retrieved_texts:
            print("No relevant context found by RAG service.")
            no_context_message = f"# No Information Found\n\nSorry, we could not find any relevant information for the topic: '{query}'. Please try a different query."
            return {"summary": no_context_message, "visualization_data": {}, "sources": []}
        
        # Combine the retrieved text chunks into a single context string
        context_str = "\n\n---\n\n".join(retrieved_texts)
        sources_formatted = ""
        for i, paper in enumerate(source_papers):
            sources_formatted += f"[{i+1}] Title: {paper.get('title', 'N/A')}, Authors: {paper.get('authors', 'N/A')}\n"

        # Constructing a prompt for the model to generate a concise answer.
        prompt = (
            f"**Role:** You are an AI **Fact Extraction Engine**. Your only task is to provide a direct and concise answer to the user's query based on the provided text.\n\n"
            f"**Core Task:** You MUST answer the user's query by extracting the most relevant information from the **CONTEXT TO USE**. You are forbidden from using external knowledge.\n\n"
            f"**Strict Rules:**\n"
            f"1. **Brevity:** The main answer MUST be **a maximum of three sentences**.\n"
            f"2. **Grounding:** Every fact in your answer must be directly supported by the **CONTEXT TO USE**.\n"
            f"3. **Citation:** You MUST add a citation marker like `[1]`, `[2]`, etc., after the information you use, corresponding to the **AVAILABLE SOURCES**.\n"
            f"4. **Missing Information:** If the context does not contain the answer, you MUST state: 'The provided context does not contain the answer.'\n"
            f"5. **References:** After your concise answer, include a 'References' section listing all cited sources.\n\n"
            f"--- --- ---\n\n"
            f"**USER QUERY:**\n"
            f"{query}\n\n"
            f"**AVAILABLE SOURCES:**\n"
            f"{sources_formatted}\n"
            f"**CONTEXT TO USE:**\n"
            f"{context_str}\n\n"
            f"--- --- ---\n\n"
            f"**Concise Answer (MAX 3 sentences):**"
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