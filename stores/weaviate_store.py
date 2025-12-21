"""
Weaviate vector store implementation for RAG.
"""

from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from loguru import logger

from config.settings import settings
from embeddings.embedder import Embedder


class WeaviateStore:
    """
    Vector store implementation using Weaviate.
    """
    
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        class_name: str = None,
        embedder: Embedder = None
    ):
        """
        Initialize Weaviate store.
        
        Args:
            url: Weaviate instance URL
            api_key: Weaviate API key (optional)
            class_name: Name of the Weaviate class
            embedder: Embedder instance for generating vectors
        """
        self.url = url or settings.WEAVIATE_URL
        self.api_key = api_key or settings.WEAVIATE_API_KEY
        self.class_name = class_name or settings.WEAVIATE_CLASS_NAME
        self.embedder = embedder or Embedder()
        
        # Initialize Weaviate client
        try:
            if self.api_key:
                self.client = weaviate.connect_to_wcs(
                    cluster_url=self.url,
                    auth_credentials=weaviate.auth.AuthApiKey(self.api_key)
                )
            else:
                self.client = weaviate.connect_to_local(
                    host=self.url.replace("http://", "").replace("https://", "")
                )
            
            logger.info(f"Connected to Weaviate at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    def create_schema(self) -> None:
        """
        Create the Weaviate schema/class for storing documents.
        """
        try:
            # Check if collection already exists
            if self.client.collections.exists(self.class_name):
                logger.info(f"Collection '{self.class_name}' already exists")
                return
            
            # Create collection with vector configuration
            logger.info(f"Creating Weaviate collection '{self.class_name}'")
            
            self.client.collections.create(
                name=self.class_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(
                        name="text",
                        data_type=DataType.TEXT,
                        description="The document text"
                    ),
                    Property(
                        name="doc_id",
                        data_type=DataType.INT,
                        description="Document ID"
                    ),
                    Property(
                        name="chunk_id",
                        data_type=DataType.INT,
                        description="Chunk ID within document"
                    )
                ]
            )
            
            logger.info(f"Successfully created collection '{self.class_name}'")
        
        except Exception as e:
            logger.error(f"Error creating Weaviate schema: {e}")
            raise
    
    def add_texts(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]] = None
    ) -> None:
        """
        Add texts to Weaviate.
        
        Args:
            texts: List of texts to add
            metadata: Optional list of metadata dicts for each text
        """
        if not texts:
            logger.warning("No texts provided to add")
            return
        
        try:
            # Ensure schema exists
            if not self.client.collections.exists(self.class_name):
                self.create_schema()
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.embedder.embed_batch(texts)
            
            # Get collection
            collection = self.client.collections.get(self.class_name)
            
            # Add documents
            with collection.batch.dynamic() as batch:
                for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                    properties = {
                        "text": text,
                        "doc_id": i,
                        "chunk_id": 0
                    }
                    
                    # Add metadata if provided
                    if metadata and i < len(metadata):
                        for key, value in metadata[i].items():
                            if key not in properties:
                                properties[key] = value
                    
                    batch.add_object(
                        properties=properties,
                        vector=embedding
                    )
            
            logger.info(f"Successfully added {len(texts)} texts to Weaviate")
        
        except Exception as e:
            logger.error(f"Error adding texts to Weaviate: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query Weaviate for similar documents.
        
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
            
            # Get collection
            collection = self.client.collections.get(self.class_name)
            
            # Query Weaviate
            logger.info(f"Querying Weaviate for: '{query_text[:50]}...'")
            
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True)
            )
            
            # Format results
            matches = []
            for obj in response.objects:
                matches.append({
                    "id": str(obj.uuid),
                    "score": 1 - obj.metadata.distance if obj.metadata.distance else 0,
                    "text": obj.properties.get("text", ""),
                    "metadata": obj.properties
                })
            
            logger.info(f"Found {len(matches)} matches in Weaviate")
            return matches
        
        except Exception as e:
            logger.error(f"Error querying Weaviate: {e}")
            raise
    
    def delete_schema(self) -> None:
        """Delete the Weaviate class/schema."""
        try:
            self.client.collections.delete(self.class_name)
            logger.info(f"Deleted Weaviate collection '{self.class_name}'")
        except Exception as e:
            logger.error(f"Error deleting Weaviate collection: {e}")
            raise
    
    def close(self) -> None:
        """Close the Weaviate client connection."""
        try:
            self.client.close()
            logger.info("Closed Weaviate connection")
        except Exception as e:
            logger.error(f"Error closing Weaviate connection: {e}")

