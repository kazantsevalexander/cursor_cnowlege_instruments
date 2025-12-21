"""
Relevance AI vector store implementation for RAG.
"""

from typing import List, Dict, Any
from relevanceai import RelevanceAI
from loguru import logger

from config.settings import settings
from embeddings.embedder import Embedder


class RelevanceStore:
    """
    Vector store implementation using Relevance AI.
    """
    
    def __init__(
        self,
        project: str = None,
        api_key: str = None,
        dataset_id: str = None,
        embedder: Embedder = None
    ):
        """
        Initialize Relevance AI store.
        
        Args:
            project: Relevance AI project ID (not used in new API)
            api_key: Relevance AI API key
            dataset_id: Dataset/collection ID
            embedder: Embedder instance for generating vectors
        """
        self.project = project or settings.RELEVANCE_PROJECT
        self.api_key = api_key or settings.RELEVANCE_API_KEY
        self.dataset_id = dataset_id or settings.RELEVANCE_DATASET_ID
        self.embedder = embedder or Embedder()
        
        if not self.api_key:
            raise ValueError("Relevance AI API key is required")
        
        # Initialize Relevance AI client (new API v10+)
        try:
            # New API requires region (default: us-east-1)
            self.client = RelevanceAI(
                api_key=self.api_key,
                project=self.project,
                region="us-east-1"  # or "eu-west-1", "ap-southeast-2"
            )
            logger.info(f"Initialized RelevanceStore with dataset: {self.dataset_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Relevance AI client: {e}")
            raise
    
    def create_collection(self) -> None:
        """
        Create a new dataset/collection in Relevance AI.
        """
        try:
            # Check if dataset exists
            datasets = self.client.list_datasets()
            
            if self.dataset_id in datasets:
                logger.info(f"Dataset '{self.dataset_id}' already exists")
                return
            
            # Create new dataset
            logger.info(f"Creating Relevance AI dataset '{self.dataset_id}'")
            self.client.create_dataset(dataset_id=self.dataset_id)
            logger.info(f"Successfully created dataset '{self.dataset_id}'")
        
        except Exception as e:
            logger.error(f"Error creating Relevance AI dataset: {e}")
            raise
    
    def add_texts(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]] = None
    ) -> None:
        """
        Add texts to Relevance AI dataset.
        
        Args:
            texts: List of texts to add
            metadata: Optional list of metadata dicts for each text
        """
        if not texts:
            logger.warning("No texts provided to add")
            return
        
        try:
            # Ensure dataset exists
            datasets = self.client.list_datasets()
            if self.dataset_id not in datasets:
                self.create_collection()
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.embedder.embed_batch(texts)
            
            # Prepare documents
            documents = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                doc = {
                    "_id": f"doc_{i}",
                    "text": text,
                    "text_vector_": embedding,
                    "doc_id": i
                }
                
                # Add metadata if provided
                if metadata and i < len(metadata):
                    doc.update(metadata[i])
                
                documents.append(doc)
            
            # Insert documents
            logger.info(f"Inserting {len(documents)} documents into Relevance AI")
            self.client.insert_documents(
                dataset_id=self.dataset_id,
                docs=documents
            )
            
            logger.info(f"Successfully added {len(texts)} texts to Relevance AI")
        
        except Exception as e:
            logger.error(f"Error adding texts to Relevance AI: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_dict: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Query Relevance AI for similar documents.
        
        Args:
            query_text: The query text
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of matching documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query_text)
            
            # Query Relevance AI
            logger.info(f"Querying Relevance AI for: '{query_text[:50]}...'")
            
            results = self.client.vector_search(
                dataset_id=self.dataset_id,
                vector=query_embedding,
                field="text_vector_",
                page_size=top_k,
                filters=filter_dict or []
            )
            
            # Format results
            matches = []
            for result in results.get("results", []):
                matches.append({
                    "id": result.get("_id", ""),
                    "score": result.get("_relevance", 0),
                    "text": result.get("text", ""),
                    "metadata": {
                        k: v for k, v in result.items() 
                        if k not in ["_id", "text", "text_vector_", "_relevance"]
                    }
                })
            
            logger.info(f"Found {len(matches)} matches in Relevance AI")
            return matches
        
        except Exception as e:
            logger.error(f"Error querying Relevance AI: {e}")
            raise
    
    def delete_collection(self) -> None:
        """Delete the Relevance AI dataset."""
        try:
            self.client.delete_dataset(dataset_id=self.dataset_id)
            logger.info(f"Deleted Relevance AI dataset '{self.dataset_id}'")
        except Exception as e:
            logger.error(f"Error deleting Relevance AI dataset: {e}")
            raise

