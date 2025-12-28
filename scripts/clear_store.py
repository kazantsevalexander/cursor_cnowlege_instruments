"""
Скрипт для очистки данных в векторных хранилищах.

Использование:
    python scripts/clear_store.py weaviate
    python scripts/clear_store.py pinecone
    python scripts/clear_store.py all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import Retriever


def main():
    if len(sys.argv) < 2:
        print("Использование: python scripts/clear_store.py <store>")
        print("  store: weaviate | pinecone | all")
        sys.exit(1)
    
    store_type = sys.argv[1].lower()
    
    if store_type not in ["weaviate", "pinecone", "all"]:
        print(f"❌ Неизвестное хранилище: {store_type}")
        sys.exit(1)
    
    print("🔧 Инициализация...")
    retriever = Retriever()
    
    try:
        if store_type == "all":
            print("\n🗑️  Очистка всех хранилищ...")
            retriever.clear_all_stores()
        else:
            print(f"\n🗑️  Очистка {store_type.upper()}...")
            retriever.clear_store(store_type)
            print(f"✅ {store_type.upper()} очищен!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        retriever.cleanup()
    
    print("\n✅ Готово!")


if __name__ == "__main__":
    main()
