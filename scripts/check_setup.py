"""
Setup verification script.
Checks that all dependencies and configurations are correct.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check if all required packages are installed."""
    print("\n" + "=" * 60)
    print("  Checking Package Installations")
    print("=" * 60 + "\n")
    
    packages = {
        "openai": "openai",
        "pinecone": "pinecone-client",
        "weaviate": "weaviate-client",
        "relevanceai": "relevanceai",
        "dotenv": "python-dotenv",
        "loguru": "loguru",
        "tiktoken": "tiktoken",
        "numpy": "numpy"
    }
    
    all_installed = True
    
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def check_configuration():
    """Check if configuration is valid."""
    print("\n" + "=" * 60)
    print("  Checking Configuration")
    print("=" * 60 + "\n")
    
    try:
        from config.settings import settings
        
        config_items = {
            "OpenAI API Key": bool(settings.OPENAI_API_KEY),
            "Pinecone API Key": bool(settings.PINECONE_API_KEY),
            "Pinecone Environment": bool(settings.PINECONE_ENVIRONMENT),
            "Weaviate URL": bool(settings.WEAVIATE_URL),
            "Relevance Project": bool(settings.RELEVANCE_PROJECT),
            "Relevance API Key": bool(settings.RELEVANCE_API_KEY),
        }
        
        all_configured = True
        
        for item, status in config_items.items():
            if status:
                print(f"  ✅ {item}")
            else:
                print(f"  ❌ {item} - NOT CONFIGURED")
                all_configured = False
        
        return all_configured
    
    except Exception as e:
        print(f"  ❌ Error loading configuration: {e}")
        return False


def check_openai_connection():
    """Test OpenAI API connection."""
    print("\n" + "=" * 60)
    print("  Testing OpenAI Connection")
    print("=" * 60 + "\n")
    
    try:
        from embeddings.embedder import Embedder
        
        embedder = Embedder()
        print("  → Generating test embedding...", end=" ")
        embedding = embedder.embed_text("test")
        
        if embedding and len(embedding) > 0:
            print("✅")
            print(f"  → Embedding dimension: {len(embedding)}")
            return True
        else:
            print("❌")
            return False
    
    except Exception as e:
        print(f"❌")
        print(f"  Error: {e}")
        return False


def check_vector_stores():
    """Check connection to vector stores."""
    print("\n" + "=" * 60)
    print("  Testing Vector Store Connections")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Test Pinecone
    try:
        from stores.pinecone_store import PineconeStore
        from embeddings.embedder import Embedder
        
        print("  → Pinecone...", end=" ")
        embedder = Embedder()
        store = PineconeStore(embedder=embedder)
        print("✅")
        results["pinecone"] = True
    except Exception as e:
        print(f"❌ ({e})")
        results["pinecone"] = False
    
    # Test Weaviate
    try:
        from stores.weaviate_store import WeaviateStore
        from embeddings.embedder import Embedder
        
        print("  → Weaviate...", end=" ")
        embedder = Embedder()
        store = WeaviateStore(embedder=embedder)
        store.close()
        print("✅")
        results["weaviate"] = True
    except Exception as e:
        print(f"❌ ({e})")
        results["weaviate"] = False
    
    # Test Relevance AI
    try:
        from stores.relevance_store import RelevanceStore
        from embeddings.embedder import Embedder
        
        print("  → Relevance AI...", end=" ")
        embedder = Embedder()
        store = RelevanceStore(embedder=embedder)
        print("✅")
        results["relevance"] = True
    except Exception as e:
        print(f"❌ ({e})")
        results["relevance"] = False
    
    return results


def print_summary(imports_ok, config_ok, openai_ok, store_results):
    """Print summary of checks."""
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60 + "\n")
    
    # Count only non-None results (skip optional checks)
    active_store_checks = [v for v in store_results.values() if v is not None]
    total_checks = 3 + len(active_store_checks)
    passed_checks = sum([imports_ok, config_ok, openai_ok]) + sum(active_store_checks)
    
    print(f"  Total checks: {total_checks}")
    print(f"  Passed: {passed_checks}")
    print(f"  Failed: {total_checks - passed_checks}")
    
    skipped = sum(1 for v in store_results.values() if v is None)
    if skipped > 0:
        print(f"  Skipped: {skipped} (optional)\n")
    else:
        print()
    
    if passed_checks == total_checks:
        print("  🎉 All checks passed! You're ready to use RAG Vector Demo.")
        print("  📝 Note: Using Pinecone + Weaviate (Relevance AI is optional)")
    elif openai_ok and any(v for v in store_results.values() if v):
        print("  ⚠️  Some checks failed, but you can still use the working stores.")
    else:
        print("  ❌ Setup incomplete. Please fix the errors above.")
        print("\n  Quick fixes:")
        if not imports_ok:
            print("    - Run: pip install -r requirements.txt")
        if not config_ok:
            print("    - Create .env file with required API keys")
        if not openai_ok:
            print("    - Verify your OpenAI API key is valid")
    
    print()


def main():
    """Run all setup checks."""
    print("\n" + "█" * 60)
    print("  RAG VECTOR DEMO - SETUP VERIFICATION")
    print("█" * 60)
    
    # Run checks
    imports_ok = check_imports()
    
    if not imports_ok:
        print("\n⚠️  Some packages are missing. Install with:")
        print("    pip install -r requirements.txt\n")
        return
    
    config_ok = check_configuration()
    openai_ok = False
    store_results = {"pinecone": False, "weaviate": False, "relevance": False}
    
    if config_ok:
        openai_ok = check_openai_connection()
        
        if openai_ok:
            store_results = check_vector_stores()
    else:
        print("\n⚠️  Configuration incomplete. Create a .env file with required keys.")
    
    # Print summary
    print_summary(imports_ok, config_ok, openai_ok, store_results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Check interrupted by user\n")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}\n")

