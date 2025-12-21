"""
Pinecone vector store implementation for RAG.
"""

from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from loguru import logger

from config.settings import settings
from embeddings.embedder import Embedder


class PineconeStore:
    """
    Vector store implementation using Pinecone.
    """
    
    def __init__(
        self,
        api_key: str = None,
        environment: str = None,
        index_name: str = None,
        embedder: Embedder = None
    ):
        """
        Initialize Pinecone store.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            embedder: Embedder instance for generating vectors
        """
        self.api_key = api_key or settings.PINECONE_API_KEY
        self.environment = environment or settings.PINECONE_ENVIRONMENT
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.embedder = embedder or Embedder()
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        
        logger.info(f"Initialized PineconeStore with index: {self.index_name}")
    
    def create_index(self, dimension: int = None, metric: str = "cosine") -> None:
        """
        Create a new Pinecone index.
        
        Args:
            dimension: Dimension of the vectors
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        dimension = dimension or self.embedder.get_embedding_dimension()
        
        try:
            # Check if index already exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name in index_names:
                logger.info(f"Index '{self.index_name}' already exists")
                self.index = self.pc.Index(self.index_name)
                return
            
            # Create new index
            logger.info(f"Creating Pinecone index '{self.index_name}' with dimension {dimension}")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Successfully created index '{self.index_name}'")
        
        except Exception as e:
            logger.error(f"Error creating Pinecone index: {e}")
            raise
    
    def add_texts(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]] = None,
        namespace: str = ""
    ) -> None:
        """
        Add texts to the Pinecone index.
        
        Args:
            texts: List of texts to add
            metadata: Optional list of metadata dicts for each text
            namespace: Pinecone namespace
        """
        if not self.index:
            self.create_index()
        
        if not texts:
            logger.warning("No texts provided to add")
            return
        
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.embedder.embed_batch(texts)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                vector_id = f"doc_{i}"
                vector_metadata = {"text": text}
                
                if metadata and i < len(metadata):
                    vector_metadata.update(metadata[i])
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": vector_metadata
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                logger.debug(f"Upserted batch {i // batch_size + 1}")
            
            logger.info(f"Successfully added {len(texts)} texts to Pinecone")
        
        except Exception as e:
            logger.error(f"Error adding texts to Pinecone: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        namespace: str = "",
        filter_dict: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the Pinecone index.
        
        Args:
            query_text: The query text
            top_k: Number of results to return
            namespace: Pinecone namespace to query
            filter_dict: Optional metadata filter
            
        Returns:
            List of matching documents with scores
        """
        if not self.index:
            logger.error("Index not initialized. Call create_index() first.")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query_text)
            
            # Query Pinecone
            logger.info(f"Querying Pinecone for: '{query_text[:50]}...'")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            matches = []
            for match in results.matches:
                matches.append({
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "metadata": match.metadata
                })
            
            logger.info(f"Found {len(matches)} matches in Pinecone")
            return matches
        
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            raise
    
    def delete_index(self) -> None:
        """Delete the Pinecone index."""
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Deleted Pinecone index '{self.index_name}'")
        except Exception as e:
            logger.error(f"Error deleting Pinecone index: {e}")
            raise

