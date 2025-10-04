import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from app.config import settings

class RAGService:
    """
    A Retrieval-Augmented Generation (RAG) service class that connects to Pinecone
    and uses a sentence transformer to retrieve relevant documents.
    This class is updated to use the modern pinecone-client v3.x syntax.
    """

    def __init__(self):
        """
        Initializes the RAGService.

        Args:
            pinecone_api_key (str): Your Pinecone API key.
            pinecone_environment (str): The Pinecone environment for your index (e.g., 'us-east-1').
        """
        if not settings.PINECONE_API_KEY:
            raise ValueError("Pinecone API key and environment must be provided.")
            
        self.pinecone_api_key = settings.PINECONE_API_KEY
        
        # The _warm_up function is called during class initialization
        self._warm_up()

    def _warm_up(self):
        """
        Initializes the necessary components for the RAG service:
        - Connects to the Pinecone index.
        - Loads the sentence transformer model.
        """
        print("Warming up RAG Service...")
        
        # --- Configuration for Pinecone and Model ---
        self.index_name = "research-papers"
        self.model_name = "all-MiniLM-L6-v2"

        # --- Initialize Pinecone (Updated) ---
        print(f"Connecting to Pinecone index: '{self.index_name}'...")
        try:
            # The modern client is initialized with the api_key.
            # The environment is no longer needed here for serverless indexes,
            # but passing it ensures compatibility with older, pod-based indexes.
            pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Get a handle to the specific index
            self.index = pc.Index(self.index_name)
            
            # A quick check to ensure the connection is valid
            print("Pinecone index stats:", self.index.describe_index_stats())
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            raise

        # --- Initialize Sentence Transformer Model ---
        print(f"Loading sentence transformer model: '{self.model_name}'...")
        try:
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading sentence transformer model: {e}")
            raise
            
        print("RAG Service is ready.")

    def query(self, user_query: str, top_k: int = 100) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Searches the Pinecone index, processes the results, and returns a combined
        list of texts and a list of unique source papers.

        Args:
            user_query (str): The query text from the user.
            top_k (int, optional): The number of results to retrieve. Defaults to 100.

        Returns:
            Tuple[List[str], List[Dict[str, str]]]: 
                A tuple containing:
                - A list of the retrieved text chunks.
                - A list of unique source papers, each as a dictionary with 'title' and 'authors'.
        """
        if not user_query:
            print("Warning: Query is empty. Returning no results.")
            return [], []
            
        print(f"Received query: '{user_query}'")
        
        try:
            # Create the vector embedding for the user's query
            query_embedding = self.model.encode(user_query).tolist()
            
            # Query Pinecone to get the most similar document chunks
            print(f"Querying Pinecone index for top {top_k} results...")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            matches = results.get('matches', [])
            print(f"Found {len(matches)} matches.")
            
            # Process the results to separate text from sources
            text_list = []
            source_papers = []
            seen_papers = set() # To track unique papers

            for match in matches:
                metadata = match.get('metadata', {})
                text = metadata.get('text')
                title = metadata.get('title')
                
                if text:
                    text_list.append(text)

                # Add paper details if the title is present and hasn't been seen before
                if title and title not in seen_papers:
                    seen_papers.add(title)
                    source_papers.append({
                        "title": title,
                        "authors": metadata.get('authors', 'N/A')
                    })
            
            return text_list, source_papers

        except Exception as e:
            print(f"An error occurred during the query process: {e}")
            return [], []

