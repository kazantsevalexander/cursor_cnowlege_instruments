"""
Embedding generation using OpenAI's text-embedding-3-large model.
"""

from typing import List
from openai import OpenAI
from loguru import logger

from config.settings import settings


class Embedder:
    """
    Generate embeddings using OpenAI's embedding models.
    """
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None
    ):
        """
        Initialize the Embedder with OpenAI client.
        
        Args:
            model: OpenAI embedding model to use (defaults to settings)
            api_key: OpenAI API key (defaults to settings)
        """
        self.model = model or settings.EMBEDDING_MODEL
        self.api_key = api_key or settings.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized Embedder with model: {self.model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of float values representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return []
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text (length: {len(text)} chars)")
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            logger.warning("All texts in batch are empty")
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            response = self.client.embeddings.create(
                input=valid_texts,
                model=self.model
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the current model.
        
        Returns:
            Dimension of the embedding vector
        """
        # Known dimensions for OpenAI models
        model_dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        
        dimension = model_dimensions.get(self.model, settings.EMBEDDING_DIMENSION)
        return dimension

