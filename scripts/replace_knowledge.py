"""
Скрипт для загрузки базы знаний в векторные хранилища.

Автоматически загружает все .txt файлы из папки knowledge/

Использование:
    python scripts/replace_knowledge.py
    python scripts/replace_knowledge.py --store pinecone
    python scripts/replace_knowledge.py --store weaviate
    python scripts/replace_knowledge.py --clear  # очистить и загрузить заново
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import Retriever

# Папка с файлами базы знаний
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"


def find_knowledge_files() -> list[Path]:
    """Находит все .txt файлы в папке knowledge/"""
    if not KNOWLEDGE_DIR.exists():
        KNOWLEDGE_DIR.mkdir(parents=True)
        print(f"📁 Создана папка: {KNOWLEDGE_DIR}")
        return []
    
    files = list(KNOWLEDGE_DIR.glob("*.txt"))
    return files


def load_files(files: list[Path]) -> tuple[list[str], list[dict]]:
    """Загружает файлы и разбивает на абзацы."""
    all_paragraphs = []
    all_metadata = []
    
    for file_path in files:
        print(f"  📄 {file_path.name}...", end=" ")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, para in enumerate(paragraphs):
            all_paragraphs.append(para)
            all_metadata.append({
                "source": file_path.name,
                "paragraph_id": i
            })
        
        print(f"{len(paragraphs)} абзацев")
    
    return all_paragraphs, all_metadata


def load_knowledge(store_type: str = "all", clear_first: bool = False):
    """Загружает базу знаний в указанные хранилища."""
    
    # Находим файлы
    print(f"\n📂 Поиск файлов в {KNOWLEDGE_DIR}/")
    files = find_knowledge_files()
    
    if not files:
        print("❌ Файлы не найдены! Поместите .txt файлы в папку knowledge/")
        return
    
    print(f"✅ Найдено {len(files)} файлов\n")
    
    # Загружаем файлы
    print("📖 Загрузка файлов:")
    paragraphs, metadata = load_files(files)
    print(f"\n📊 Всего: {len(paragraphs)} абзацев из {len(files)} файлов")
    
    # Инициализируем retriever
    print("\n🔧 Инициализация RAG системы...")
    retriever = Retriever()
    
    # Определяем хранилища
    if store_type == "all":
        stores = ["pinecone", "weaviate", "relevance"]
    else:
        stores = [store_type]
    
    # Загружаем данные
    for store in stores:
        print(f"\n{'='*60}")
        print(f"📤 Загрузка в {store.upper()}...")
        print(f"{'='*60}")
        
        try:
            if clear_first:
                print("  🗑️  Очистка старых данных...")
                retriever.clear_store(store)
            
            retriever.add_documents(
                texts=paragraphs,
                store_type=store,
                metadata=metadata
            )
            print(f"✅ {store.upper()}: загружено {len(paragraphs)} документов!")
        except Exception as e:
            print(f"❌ {store.upper()}: ошибка - {e}")
    
    retriever.cleanup()
    print("\n✅ Готово!")


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка базы знаний из папки knowledge/"
    )
    parser.add_argument(
        "--store", "-s",
        choices=["pinecone", "weaviate", "relevance", "all"],
        default="all",
        help="Хранилище для загрузки (по умолчанию: all)"
    )
    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Очистить хранилище перед загрузкой"
    )
    
    args = parser.parse_args()
    load_knowledge(args.store, args.clear)


if __name__ == "__main__":
    main()
