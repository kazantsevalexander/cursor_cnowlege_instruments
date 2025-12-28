"""
Тестовый скрипт для демонстрации полного цикла RAG:
1. Загрузка файла
2. Создание эмбеддингов
3. Отправка в векторные БД (Pinecone и Weaviate)
4. Тестовый запрос
5. Вывод найденных результатов
"""

import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from rag.retriever import Retriever
from embeddings.embedder import Embedder


def load_all_files(knowledge_dir: Path) -> tuple[list[str], list[dict]]:
    """
    Загружает все .txt файлы из папки и возвращает абзацы с метаданными.
    
    Args:
        knowledge_dir: Путь к папке с файлами
        
    Returns:
        Кортеж (список абзацев, список метаданных)
    """
    txt_files = list(knowledge_dir.glob("*.txt"))
    
    if not txt_files:
        return [], []
    
    all_paragraphs = []
    all_metadata = []
    
    for txt_file in txt_files:
        print(f"  📄 {txt_file.name}...", end=" ")
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, para in enumerate(paragraphs):
            all_paragraphs.append(para)
            all_metadata.append({
                "source": txt_file.name,
                "paragraph_id": i
            })
        
        print(f"{len(paragraphs)} абзацев")
    
    return all_paragraphs, all_metadata


def create_embeddings_and_store(paragraphs: list[str], metadata: list[dict], retriever: Retriever):
    """
    Создает эмбеддинги и сохраняет в векторные БД.
    
    Args:
        paragraphs: Список текстов для обработки
        metadata: Список метаданных для каждого абзаца
        retriever: RAG retriever
    """
    print("=" * 80)
    print("📊 ШАГ 1: СОЗДАНИЕ ЭМБЕДДИНГОВ И ОТПРАВКА В ВЕКТОРНЫЕ БД")
    print("=" * 80 + "\n")
    
    # Список векторных БД для использования
    stores = ["pinecone", "weaviate"]
    
    for store_type in stores:
        try:
            print(f"🔄 Обработка {store_type.upper()}...")
            print(f"   → Создание эмбеддингов для {len(paragraphs)} текстов...")
            
            # Добавляем документы (эмбеддинги создаются автоматически внутри)
            retriever.add_documents(
                texts=paragraphs,
                store_type=store_type,
                metadata=metadata
            )
            
            print(f"   ✅ {store_type.upper()}: {len(paragraphs)} документов успешно добавлено\n")
            
        except Exception as e:
            print(f"   ❌ Ошибка при работе с {store_type}: {e}\n")


def test_query(query: str, retriever: Retriever, top_k: int = 3):
    """
    Выполняет тестовый запрос и выводит результаты.
    
    Args:
        query: Текст запроса
        retriever: RAG retriever
        top_k: Количество результатов
    """
    print("\n" + "=" * 80)
    print("🔍 ШАГ 2: ВЫПОЛНЕНИЕ ТЕСТОВОГО ЗАПРОСА")
    print("=" * 80 + "\n")
    
    print(f"❓ ЗАПРОС: {query}\n")
    
    stores = ["pinecone", "weaviate"]
    
    for store_type in stores:
        try:
            print("─" * 80)
            print(f"📊 РЕЗУЛЬТАТЫ ИЗ {store_type.upper()}")
            print("─" * 80 + "\n")
            
            # Выполняем поиск
            results = retriever.retrieve(
                query=query,
                store_type=store_type,
                top_k=top_k
            )
            
            if not results:
                print("  ⚠️  Результаты не найдены\n")
                continue
            
            # Выводим результаты
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                text = result.get('text', '')
                metadata = result.get('metadata', {})
                
                print(f"  [{i}] 📈 Релевантность: {score:.4f}")
                
                # Если текст длинный - обрезаем для вывода
                if len(text) > 200:
                    text_preview = text[:200] + "..."
                else:
                    text_preview = text
                
                print(f"      📝 Текст: {text_preview}")
                
                # Выводим метаданные если есть
                if metadata:
                    para_id = metadata.get('paragraph_id', 'N/A')
                    source = metadata.get('source', 'N/A')
                    print(f"      📌 Метаданные: Абзац #{para_id}, Источник: {source}")
                
                print()
            
        except Exception as e:
            print(f"  ❌ Ошибка при запросе к {store_type}: {e}\n")


def compare_results(query: str, retriever: Retriever, top_k: int = 3):
    """
    Сравнивает результаты из разных векторных БД.
    
    Args:
        query: Текст запроса
        retriever: RAG retriever
        top_k: Количество результатов
    """
    print("\n" + "=" * 80)
    print("📊 ШАГ 3: СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80 + "\n")
    
    print(f"❓ ЗАПРОС: {query}\n")
    
    # Получаем результаты из всех хранилищ
    all_results = retriever.retrieve_all(query, top_k=top_k)
    
    # Сравниваем первые результаты
    print("🏆 ЛУЧШИЙ РЕЗУЛЬТАТ ИЗ КАЖДОЙ БД:\n")
    
    for store_type, results in all_results.items():
        if results and len(results) > 0:
            best = results[0]
            score = best.get('score', 0)
            text = best.get('text', '')[:100]
            
            print(f"  {store_type.upper():12} → Score: {score:.4f} | {text}...")
        else:
            print(f"  {store_type.upper():12} → Нет результатов")
    
    print()


def main():
    """Главная функция."""
    
    print("\n" + "█" * 80)
    print("  🧪 ТЕСТИРОВАНИЕ RAG СИСТЕМЫ")
    print("█" * 80 + "\n")
    
    # Папка с файлами базы знаний
    knowledge_dir = Path(__file__).parent / "knowledge"
    
    try:
        # 1. Загружаем все файлы из папки knowledge/
        if not knowledge_dir.exists():
            print(f"❌ Папка '{knowledge_dir}' не найдена!")
            print(f"   Создайте папку knowledge/ и поместите туда .txt файлы.\n")
            return
        
        txt_files = list(knowledge_dir.glob("*.txt"))
        if not txt_files:
            print(f"❌ В папке '{knowledge_dir}' нет .txt файлов!")
            return
        
        print(f"📂 Найдено {len(txt_files)} файлов в knowledge/\n")
        
        # Загружаем все файлы
        paragraphs, metadata = load_all_files(knowledge_dir)
        
        if not paragraphs:
            print("❌ Не удалось загрузить абзацы из файлов!")
            return
        
        print(f"\n📊 Всего загружено: {len(paragraphs)} абзацев\n")
        
        # Выводим превью первого абзаца
        print("📄 Превью первого абзаца:")
        print("   " + paragraphs[0][:150] + "...\n")
        
        # 2. Инициализируем RAG систему
        print("🔧 Инициализация RAG системы...")
        retriever = Retriever()
        print("✅ RAG система инициализирована\n")
        
        # 3. Создаем эмбеддинги и отправляем в БД
        create_embeddings_and_store(paragraphs, metadata, retriever)
        
        # 4. Тестовые запросы
        test_queries = [
            "К какому классу сложности относится здание высотой 35 метров?",
            "Что относится к зданиям первого класса сложности К-1?",
            "Какой класс сложности у подстанций напряжением 110 кВ?",
            "К какому классу относятся мосты длиной более 100 метров?",
            "Что входит в здания и сооружения третьего класса сложности?",
            "Какой класс сложности у одноквартирного жилого дома до 7 метров?",
            "К какому классу относятся производственные здания площадью 6000 м2?",
            "Чем отличается К-2 от К-3 по типам объектов?"
        ]
        
        # Выполняем первый запрос детально
        test_query(test_queries[0], retriever, top_k=3)
        
        # Сравниваем результаты для другого запроса
        compare_results(test_queries[2], retriever, top_k=2)
        
        # 5. Интерактивный режим (опционально)
        print("\n" + "=" * 80)
        print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ (для выхода введите 'exit')")
        print("=" * 80 + "\n")
        
        while True:
            try:
                user_query = input("\n❓ Ваш запрос: ").strip()
                
                if user_query.lower() in ['exit', 'quit', 'выход']:
                    print("\n👋 Выход из программы...\n")
                    break
                
                if not user_query:
                    continue
                
                # Выполняем запрос
                test_query(user_query, retriever, top_k=3)
                
            except KeyboardInterrupt:
                print("\n\n👋 Выход из программы...\n")
                break
        
        # Завершение
        print("=" * 80)
        print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 80 + "\n")
        
        retriever.cleanup()
        
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        print(f"   Поместите .txt файлы в папку knowledge/\n")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

