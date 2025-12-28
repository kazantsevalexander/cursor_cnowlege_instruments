"""
Configuration module for RAG Vector Demo project.
Loads environment variables and provides centralized settings.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Centralized configuration settings for the RAG application."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rag-demo-index")
    
    # Weaviate Configuration
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_API_KEY: Optional[str] = os.getenv("WEAVIATE_API_KEY")
    WEAVIATE_CLASS_NAME: str = "RAGDocument"
    
    # Relevance AI Configuration
    RELEVANCE_PROJECT: str = os.getenv("RELEVANCE_PROJECT", "")
    RELEVANCE_API_KEY: str = os.getenv("RELEVANCE_API_KEY", "")
    RELEVANCE_DATASET_ID: str = os.getenv("RELEVANCE_DATASET_ID", "rag-demo-dataset")
    
    @classmethod
    def validate(cls) -> bool:
        """ Проверяет наличие всех обязательных значений конфигурации. 
        Возвращает: bool: True, если все обязательные настройки присутствуют, 
        иначе False. 
        """

        required_fields = {
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
            "PINECONE_API_KEY": cls.PINECONE_API_KEY,
            "PINECONE_ENVIRONMENT": cls.PINECONE_ENVIRONMENT,
            "WEAVIATE_URL": cls.WEAVIATE_URL,
            "RELEVANCE_PROJECT": cls.RELEVANCE_PROJECT,
            "RELEVANCE_API_KEY": cls.RELEVANCE_API_KEY,
        }
        
        missing_fields = [
            field for field, value in required_fields.items() 
            if not value
        ]
        
        if missing_fields:
            print(f"⚠️  Missing required configuration fields: {', '.join(missing_fields)}")
            return False
        
        return True
    
    @classmethod
    def print_config(cls) -> None:
        """Вывести текущую конфигурацию (без раскрытия конфиденциальных данных)."""
        print("=" * 60)
        print("RAG Vector Demo - Configuration")
        print("=" * 60)
        print(f"OpenAI Model: {cls.EMBEDDING_MODEL}")
        print(f"Embedding Dimension: {cls.EMBEDDING_DIMENSION}")
        print(f"Pinecone Index: {cls.PINECONE_INDEX_NAME}")
        print(f"Weaviate URL: {cls.WEAVIATE_URL}")
        print(f"Weaviate Class: {cls.WEAVIATE_CLASS_NAME}")
        print(f"Relevance Project: {cls.RELEVANCE_PROJECT}")
        print(f"Relevance Dataset: {cls.RELEVANCE_DATASET_ID}")
        print("=" * 60)


# Create a singleton instance
settings = Settings()

