"""
Advanced usage examples for RAG Vector Demo.
Demonstrates more complex scenarios and features.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from embeddings.embedder import Embedder
from rag.retriever import Retriever
from utils.chunker import TextChunker


def example_chunking_large_document():
    """Demonstrate chunking of a large document."""
    print("\n" + "=" * 80)
    print("Example 1: Chunking Large Document")
    print("=" * 80 + "\n")
    
    # Simulate a large document
    large_document = """
    Artificial Intelligence (AI) has become one of the most transformative 
    technologies of the 21st century. It encompasses various subfields including 
    machine learning, natural language processing, computer vision, and robotics.
    
    Machine Learning is a subset of AI that focuses on the development of algorithms 
    that can learn from and make predictions or decisions based on data. It includes 
    techniques such as supervised learning, unsupervised learning, and reinforcement 
    learning. Deep learning, a subset of machine learning, uses neural networks with 
    multiple layers to model complex patterns in data.
    
    Natural Language Processing (NLP) enables computers to understand, interpret, 
    and generate human language. Applications of NLP include machine translation, 
    sentiment analysis, chatbots, and text summarization. Recent advances in NLP, 
    particularly with transformer models like GPT and BERT, have revolutionized 
    the field.
    
    Computer Vision allows machines to interpret and understand visual information 
    from the world. It involves tasks such as image classification, object detection, 
    facial recognition, and image segmentation. Deep learning has significantly 
    improved the accuracy of computer vision systems.
    
    The ethical implications of AI are increasingly important. Issues such as bias 
    in algorithms, privacy concerns, job displacement, and the potential misuse of 
    AI technologies require careful consideration and regulation.
    """ * 3  # Repeat to make it longer
    
    # Initialize chunker
    chunker = TextChunker(chunk_size=256, chunk_overlap=30)
    
    # Chunk the document
    chunks = chunker.chunk_text(large_document)
    
    print(f"Original document length: {len(large_document)} characters")
    print(f"Number of chunks created: {len(chunks)}")
    print(f"\nFirst chunk preview:")
    print("-" * 80)
    print(chunks[0][:200] + "...")
    print("-" * 80)
    print(f"\nLast chunk preview:")
    print("-" * 80)
    print(chunks[-1][:200] + "...")
    print("-" * 80)


def example_metadata_filtering():
    """Demonstrate adding and filtering by metadata."""
    print("\n" + "=" * 80)
    print("Example 2: Metadata Filtering")
    print("=" * 80 + "\n")
    
    retriever = Retriever()
    
    # Documents with metadata
    documents = [
        "Python is a high-level programming language.",
        "JavaScript is used for web development.",
        "Java is a popular enterprise programming language.",
        "Deep learning uses neural networks.",
        "Machine learning is a subset of AI."
    ]
    
    metadata = [
        {"category": "programming", "language": "python", "year": 1991},
        {"category": "programming", "language": "javascript", "year": 1995},
        {"category": "programming", "language": "java", "year": 1995},
        {"category": "ai", "topic": "deep_learning", "year": 2010},
        {"category": "ai", "topic": "machine_learning", "year": 1950}
    ]
    
    print("Adding documents with metadata to Pinecone...")
    try:
        retriever.add_documents(
            texts=documents,
            store_type="pinecone",
            metadata=metadata
        )
        
        print("✅ Documents added successfully!\n")
        
        # Query without filter
        print("Query: 'programming languages'")
        print("\nResults without filter:")
        results = retriever.retrieve("programming languages", "pinecone", top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] {r['text'][:60]}...")
        
        # Query with metadata filter (Pinecone supports this)
        print("\nResults with filter (category='ai'):")
        results = retriever.retrieve(
            "programming languages",
            "pinecone",
            top_k=3,
            filter_dict={"category": "ai"}
        )
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] {r['text'][:60]}...")
    
    except Exception as e:
        logger.error(f"Error in metadata filtering example: {e}")
        print(f"❌ Error: {e}")


def example_batch_embedding():
    """Demonstrate efficient batch embedding."""
    print("\n" + "=" * 80)
    print("Example 3: Batch Embedding Efficiency")
    print("=" * 80 + "\n")
    
    import time
    
    embedder = Embedder()
    
    texts = [
        "Machine learning is fascinating.",
        "Deep learning powers modern AI.",
        "Natural language processing is evolving.",
        "Computer vision enables visual recognition.",
        "Reinforcement learning teaches through rewards."
    ]
    
    # Individual embeddings (slower)
    print("Generating embeddings individually...")
    start = time.time()
    individual_embeddings = [embedder.embed_text(text) for text in texts]
    individual_time = time.time() - start
    print(f"Time taken: {individual_time:.2f} seconds")
    
    # Batch embeddings (faster)
    print("\nGenerating embeddings in batch...")
    start = time.time()
    batch_embeddings = embedder.embed_batch(texts)
    batch_time = time.time() - start
    print(f"Time taken: {batch_time:.2f} seconds")
    
    speedup = individual_time / batch_time if batch_time > 0 else float('inf')
    print(f"\n📊 Batch processing is {speedup:.2f}x faster!")
    
    # Verify embeddings are similar
    print(f"\nEmbedding dimension: {len(batch_embeddings[0])}")
    print(f"Number of embeddings: {len(batch_embeddings)}")


def example_cross_store_comparison():
    """Compare the same query across all three stores."""
    print("\n" + "=" * 80)
    print("Example 4: Cross-Store Result Comparison")
    print("=" * 80 + "\n")
    
    retriever = Retriever()
    
    # Sample documents
    docs = [
        "Vector databases are optimized for similarity search.",
        "Embeddings represent text as high-dimensional vectors.",
        "RAG combines retrieval with text generation.",
        "Semantic search finds meaning, not just keywords.",
        "Dense vectors capture semantic relationships."
    ]
    
    print("Adding documents to all stores...\n")
    
    for store_type in ["pinecone", "weaviate", "relevance"]:
        try:
            print(f"  → {store_type.capitalize()}...", end=" ")
            retriever.add_documents(docs, store_type=store_type)
            print("✅")
        except Exception as e:
            print(f"❌ ({e})")
    
    # Compare results
    query = "How do vector databases work?"
    print(f"\n{'=' * 80}")
    print(f"Query: '{query}'")
    print('=' * 80)
    
    retriever.compare_stores(query, top_k=3)


def example_custom_embedding_dimensions():
    """Show how to work with different embedding models."""
    print("\n" + "=" * 80)
    print("Example 5: Embedding Dimensions")
    print("=" * 80 + "\n")
    
    # Different OpenAI models have different dimensions
    models = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
    }
    
    for model, expected_dim in models.items():
        try:
            embedder = Embedder(model=model)
            embedding = embedder.embed_text("Test text")
            actual_dim = len(embedding)
            
            print(f"Model: {model}")
            print(f"  Expected dimension: {expected_dim}")
            print(f"  Actual dimension: {actual_dim}")
            print(f"  Status: {'✅' if actual_dim == expected_dim else '❌'}\n")
        except Exception as e:
            print(f"Model: {model}")
            print(f"  Status: ❌ Error - {e}\n")


def main():
    """Run all advanced examples."""
    print("\n" + "█" * 80)
    print("  RAG VECTOR DEMO - ADVANCED USAGE EXAMPLES")
    print("█" * 80)
    
    try:
        # Run examples
        example_chunking_large_document()
        example_batch_embedding()
        example_custom_embedding_dimensions()
        
        # These require valid API credentials
        print("\n" + "=" * 80)
        print("The following examples require valid API credentials:")
        print("=" * 80)
        
        from config.settings import settings
        if settings.validate():
            example_metadata_filtering()
            example_cross_store_comparison()
        else:
            print("\n⚠️  Skipping API-dependent examples. Configure .env file first.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        logger.error(f"Error in examples: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    
    print("\n" + "█" * 80)
    print("  EXAMPLES COMPLETED")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()

