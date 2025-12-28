"""
Embedding generation using OpenAI's text-embedding-3-large model.
"""

from typing import List
from openai import OpenAI
from loguru import logger

from config.settings import settings


class Embedder:
    """
    Генерируйте эмбеддинги с помощью моделей эмбеддингов OpenAI.
    """
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None
    ):
        """
        Инициализирует Embedder с клиентом OpenAI.

        Аргументы:
            model: Модель эмбеддингов OpenAI для использования (по умолчанию из настроек)
            api_key: Ключ API OpenAI (по умолчанию из настроек)
        """
        self.model = model or settings.EMBEDDING_MODEL
        self.api_key = api_key or settings.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized Embedder with model: {self.model}")
    
    def embed_text(self, text: str) -> List[float]:
    """
    Генерирует эмбеддинг для одного текста.

    Аргументы:
        text: Текст для создания эмбеддинга

    Возвращает:
        Список значений float, представляющих вектор эмбеддинга
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
    Генерирует эмбеддинги для пакета текстов.

    Аргументы:
        texts: Список текстов для получения эмбеддингов

    Возвращает:
        Список векторов эмбеддингов
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
        Получить размерность эмбеддингов, производимых текущей моделью.

        Возвращает:
            Размерность вектора эмбеддинга

        """
        # Known dimensions for OpenAI models
        model_dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        
        dimension = model_dimensions.get(self.model, settings.EMBEDDING_DIMENSION)
        return dimension

