"""
Demo script showing RAG with three vector stores: Relevance AI, Weaviate, and Pinecone.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import settings
from embeddings.embedder import Embedder
from rag.retriever import Retriever
from utils.chunker import TextChunker


# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    """
    Machine learning is a subset of artificial intelligence that focuses on the development 
    of algorithms and statistical models that enable computers to improve their performance 
    on a specific task through experience. It involves training models on data to make 
    predictions or decisions without being explicitly programmed to do so.
    """,
    """
    Natural language processing (NLP) is a branch of artificial intelligence that helps 
    computers understand, interpret, and manipulate human language. NLP draws from many 
    disciplines, including computer science and computational linguistics, in its pursuit 
    to fill the gap between human communication and computer understanding.
    """,
    """
    Deep learning is a type of machine learning based on artificial neural networks with 
    multiple layers. These neural networks attempt to simulate the behavior of the human 
    brain, allowing it to "learn" from large amounts of data. While a neural network with 
    a single layer can make approximate predictions, additional hidden layers can optimize 
    and refine for accuracy.
    """,
    """
    Computer vision is an interdisciplinary field that deals with how computers can gain 
    high-level understanding from digital images or videos. From the perspective of 
    engineering, it seeks to automate tasks that the human visual system can do. Computer 
    vision tasks include methods for acquiring, processing, analyzing and understanding 
    digital images.
    """,
    """
    Reinforcement learning is an area of machine learning concerned with how intelligent 
    agents ought to take actions in an environment in order to maximize the notion of 
    cumulative reward. It is employed by various software and machines to find the best 
    possible behavior or path it should take in a specific situation.
    """,
    """
    Transfer learning is a machine learning method where a model developed for a task is 
    reused as the starting point for a model on a second task. It is a popular approach 
    in deep learning where pre-trained models are used as the starting point on computer 
    vision and natural language processing tasks.
    """,
    """
    Retrieval-Augmented Generation (RAG) is a technique that combines the power of 
    large language models with information retrieval systems. It works by first retrieving 
    relevant documents from a knowledge base and then using those documents to generate 
    more accurate and contextual responses. This approach helps reduce hallucinations and 
    provides more factual answers.
    """,
    """
    Vector databases are specialized databases designed to store and query high-dimensional 
    vectors efficiently. They are crucial for applications like semantic search, 
    recommendation systems, and RAG systems. Popular vector databases include Pinecone, 
    Weaviate, and Relevance AI, each offering unique features for managing embeddings.
    """
]


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def main():
    """Main demo function."""
    
    print_header("🚀 RAG Vector Store Demo")
    
    # Validate configuration
    if not settings.validate():
        logger.error("Configuration validation failed. Please check your .env file.")
        return
    
    settings.print_config()
    
    # Initialize components
    logger.info("Initializing embedder and retriever...")
    embedder = Embedder()
    retriever = Retriever(embedder=embedder)
    
    # Optional: Initialize chunker for text processing
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    
    print_header("📚 Adding Documents to Vector Stores")
    
    # Add documents to each store
    stores = ["pinecone", "weaviate", "relevance"]
    
    for store_type in stores:
        try:
            logger.info(f"Adding documents to {store_type}...")
            print(f"\n  📥 Adding {len(SAMPLE_DOCUMENTS)} documents to {store_type.upper()}...")
            
            retriever.add_documents(
                texts=SAMPLE_DOCUMENTS,
                store_type=store_type,
                metadata=[{"doc_index": i} for i in range(len(SAMPLE_DOCUMENTS))]
            )
            
            print(f"  ✅ Successfully added documents to {store_type.upper()}")
        
        except Exception as e:
            logger.error(f"Failed to add documents to {store_type}: {e}")
            print(f"  ❌ Failed to add documents to {store_type.upper()}: {e}")
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does natural language processing work?",
        "Explain retrieval-augmented generation",
        "What are vector databases used for?"
    ]
    
    print_header("🔍 Querying Vector Stores")
    
    for query in test_queries:
        try:
            retriever.compare_stores(query, top_k=3)
        except Exception as e:
            logger.error(f"Error during comparison: {e}")
    
    # Individual store queries
    print_header("🎯 Individual Store Query Examples")
    
    query = "What is deep learning?"
    print(f"Query: '{query}'\n")
    
    for store_type in stores:
        try:
            results = retriever.retrieve(query, store_type, top_k=2)
            
            print(f"\n{store_type.upper()}:")
            for i, result in enumerate(results, 1):
                print(f"  [{i}] Score: {result['score']:.4f}")
                print(f"      Text preview: {result['text'][:80]}...")
        
        except Exception as e:
            logger.error(f"Error querying {store_type}: {e}")
            print(f"\n{store_type.upper()}: Error - {e}")
    
    # Demonstrate text chunking
    print_header("✂️  Text Chunking Demo")
    
    long_text = " ".join(SAMPLE_DOCUMENTS[:3])
    chunks = chunker.chunk_text(long_text)
    
    print(f"Original text length: {len(long_text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print(f"\nFirst chunk preview:")
    print(f"  {chunks[0][:150]}...")
    
    # Cleanup
    print_header("🧹 Cleanup")
    retriever.cleanup()
    logger.info("Demo completed successfully!")
    
    print("\n✨ Demo finished! Check the logs for detailed information.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        logger.warning("Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)

