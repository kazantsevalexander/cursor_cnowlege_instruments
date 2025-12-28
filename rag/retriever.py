"""
RAG Retriever that can switch between different vector stores.
"""

from typing import List, Dict, Any, Optional, Literal
from loguru import logger

from embeddings.embedder import Embedder
from stores.pinecone_store import PineconeStore
from stores.weaviate_store import WeaviateStore

# Relevance AI is optional (may have installation issues on Windows)
try:
    from stores.relevance_store import RelevanceStore
    RELEVANCE_AVAILABLE = True
except ImportError:
    RELEVANCE_AVAILABLE = False
    logger.warning("Relevance AI not available - install separately if needed")


StoreType = Literal["pinecone", "weaviate", "relevance"]


class Retriever:
    """
    Unified retriever that can work with multiple vector stores.
    """
    
    def __init__(self, embedder: Embedder = None):
        """
        Initialize the Retriever.
        
        Args:
            embedder: Embedder instance (shared across all stores)
        """
        self.embedder = embedder or Embedder()
        
        # Initialize stores lazily
        self._stores: Dict[str, Any] = {}
        
        logger.info("Initialized Retriever with multi-store support")
    
    def _get_store(self, store_type: StoreType):
        """
        Get or initialize a vector store.
        
        Args:
            store_type: Type of store to get
            
        Returns:
            The vector store instance
        """
        if store_type not in self._stores:
            logger.info(f"Initializing {store_type} store")
            
            if store_type == "pinecone":
                self._stores[store_type] = PineconeStore(embedder=self.embedder)
            elif store_type == "weaviate":
                self._stores[store_type] = WeaviateStore(embedder=self.embedder)
            elif store_type == "relevance":
                if not RELEVANCE_AVAILABLE:
                    raise ImportError(
                        "Relevance AI is not installed. "
                        "It may have dependency conflicts on Windows. "
                        "Use Pinecone or Weaviate instead."
                    )
                self._stores[store_type] = RelevanceStore(embedder=self.embedder)
            else:
                raise ValueError(f"Unknown store type: {store_type}")
        
        return self._stores[store_type]
    
    def add_documents(
        self,
        texts: List[str],
        store_type: StoreType,
        metadata: List[Dict[str, Any]] = None
    ) -> None:
        """
        Add documents to a specific vector store.
        
        Args:
            texts: List of document texts
            store_type: Which store to use
            metadata: Optional metadata for each document
        """
        try:
            store = self._get_store(store_type)
            
            # Initialize store/index if needed
            if store_type == "pinecone":
                store.create_index()
            elif store_type == "weaviate":
                store.create_schema()
            elif store_type == "relevance":
                store.create_collection()
            
            # Add texts
            logger.info(f"Adding {len(texts)} documents to {store_type}")
            store.add_texts(texts, metadata)
            logger.info(f"Successfully added documents to {store_type}")
        
        except Exception as e:
            logger.error(f"Error adding documents to {store_type}: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        store_type: StoreType,
        top_k: int = 8,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents from a specific vector store.
        
        Args:
            query: Query text
            store_type: Which store to query
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of matching documents with scores
        """
        try:
            store = self._get_store(store_type)
            
            logger.info(f"Retrieving from {store_type}: '{query[:50]}...'")
            results = store.query(query, top_k, filter_dict)
            
            # Add store type to results
            for result in results:
                result["store"] = store_type
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving from {store_type}: {e}")
            raise
    
    def retrieve_all(
        self,
        query: str,
        top_k: int = 5,
        stores: Optional[List[StoreType]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve documents from all specified stores.
        
        Args:
            query: Query text
            top_k: Number of results per store
            stores: List of stores to query (default: all)
            
        Returns:
            Dictionary mapping store type to results
        """
        if stores is None:
            stores = ["pinecone", "weaviate"]
        
        results = {}
        
        for store_type in stores:
            try:
                results[store_type] = self.retrieve(query, store_type, top_k)
            except Exception as e:
                logger.error(f"Failed to retrieve from {store_type}: {e}")
                results[store_type] = []
        
        return results
    
    def compare_stores(
        self,
        query: str,
        top_k: int = 5
    ) -> None:
        """
        Compare results from all stores and print comparison.
        
        Args:
            query: Query text
            top_k: Number of results per store
        """
        logger.info(f"Comparing stores for query: '{query}'")
        
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)
        
        all_results = self.retrieve_all(query, top_k)
        
        for store_type, results in all_results.items():
            print(f"\n{'─' * 80}")
            print(f"📊 {store_type.upper()} RESULTS")
            print(f"{'─' * 80}")
            
            if not results:
                print("  ⚠️  No results found")
                continue
            
            for i, result in enumerate(results, 1):
                score = result.get("score", 0)
                text = result.get("text", "")
                text_preview = text[:100] + "..." if len(text) > 100 else text
                
                print(f"\n  [{i}] Score: {score:.4f}")
                print(f"      Text: {text_preview}")
        
        print("\n" + "=" * 80 + "\n")
    
    def clear_store(self, store_type: StoreType) -> None:
        """
        Clear all data from a specific vector store.
        
        Args:
            store_type: Which store to clear
        """
        try:
            store = self._get_store(store_type)
            
            logger.info(f"Clearing all data from {store_type}")
            
            if store_type == "pinecone":
                store.delete_index()
            elif store_type == "weaviate":
                store.delete_schema()
            elif store_type == "relevance":
                store.delete_collection()
            
            # Remove from cache to force re-initialization
            if store_type in self._stores:
                del self._stores[store_type]
            
            logger.info(f"Successfully cleared {store_type}")
        
        except Exception as e:
            logger.error(f"Error clearing {store_type}: {e}")
            raise
    
    def clear_all_stores(self) -> None:
        """Clear all data from all vector stores."""
        stores = ["pinecone", "weaviate", "relevance"]
        
        for store_type in stores:
            try:
                self.clear_store(store_type)
            except Exception as e:
                logger.warning(f"Could not clear {store_type}: {e}")
    
    def replace_knowledge_base(
        self,
        texts: List[str],
        store_type: StoreType,
        metadata: List[Dict[str, Any]] = None
    ) -> None:
        """
        Replace all data in a store with new documents.
        
        Args:
            texts: New document texts
            store_type: Which store to update
            metadata: Optional metadata for each document
        """
        logger.info(f"Replacing knowledge base in {store_type}")
        
        # Clear existing data
        self.clear_store(store_type)
        
        # Add new documents
        self.add_documents(texts, store_type, metadata)
        
        logger.info(f"Knowledge base replaced in {store_type}")
    
    def cleanup(self) -> None:
        """Clean up all store connections."""
        for store_type, store in self._stores.items():
            try:
                if store_type == "weaviate" and hasattr(store, 'close'):
                    store.close()
                logger.info(f"Cleaned up {store_type} store")
            except Exception as e:
                logger.error(f"Error cleaning up {store_type}: {e}")

