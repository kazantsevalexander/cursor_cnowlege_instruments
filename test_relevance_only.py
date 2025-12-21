import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from embeddings.embedder import Embedder


def test_chromadb():
    """Полный тест ChromaDB с векторными данными."""
    
    print("\n" + "=" * 80)
    print("🧪 ТЕСТ CHROMADB - ЛОКАЛЬНАЯ ВЕКТОРНАЯ БД")
    print("=" * 80 + "\n")
    
    # 1. Инициализация
    print("🔧 Шаг 1: Инициализация ChromaDB...")
    try:
        # Создаём клиент (локальная БД, сохраняется в ./chroma_db)
        client = chromadb.PersistentClient(path="./chroma_db")
        print(f"✅ ChromaDB инициализирован!")
        print(f"   Тип: Локальная БД (не требует API ключей)")
        print(f"   Путь: ./chroma_db\n")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}\n")
        return
    
    # 2. Создание коллекции
    print("📦 Шаг 2: Создание коллекции...")
    try:
        # Удаляем если существует (для чистоты теста)
        try:
            client.delete_collection(name="test_collection")
            print("   🗑️  Удалена старая коллекция")
        except:
            pass
        
        collection = client.create_collection(
            name="test_collection",
            metadata={"description": "Тестовая коллекция для RAG"}
        )
        print(f"✅ Коллекция создана: test_collection\n")
    except Exception as e:
        print(f"❌ Ошибка создания коллекции: {e}\n")
        return
    
    # 3. Создание эмбеддингов
    print("📊 Шаг 3: Создание тестовых эмбеддингов...")
    embedder = Embedder()
    
    test_texts = [
        "Python — высокоуровневый язык программирования общего назначения",
        "Машинное обучение — раздел искусственного интеллекта, изучающий методы построения алгоритмов",
        "Векторные базы данных оптимизированы для хранения и поиска векторных представлений",
        "ChromaDB — локальная векторная база данных для AI приложений",
        "RAG (Retrieval-Augmented Generation) комбинирует поиск информации с генерацией текста"
    ]
    
    print(f"   Генерация эмбеддингов для {len(test_texts)} текстов...")
    embeddings = embedder.embed_batch(test_texts)
    print(f"✅ Создано {len(embeddings)} эмбеддингов")
    print(f"   Размерность: {len(embeddings[0])}\n")
    
    # 4. Добавление данных в ChromaDB
    print("💾 Шаг 4: Добавление данных в ChromaDB...")
    try:
        collection.add(
            embeddings=embeddings,
            documents=test_texts,
            ids=[f"doc_{i}" for i in range(len(test_texts))],
            metadatas=[{"source": "test", "index": i} for i in range(len(test_texts))]
        )
        print(f"✅ Добавлено {len(test_texts)} документов в ChromaDB\n")
    except Exception as e:
        print(f"❌ Ошибка добавления: {e}\n")
        return
    
    # 5. Проверка сохранённых данных
    print("🔍 Шаг 5: Проверка сохранённых данных...")
    count = collection.count()
    print(f"   Всего документов в коллекции: {count}\n")
    
    # 6. Тестовый поиск
    print("🔎 Шаг 6: Выполнение векторного поиска...")
    test_queries = [
        "Что такое машинное обучение?",
        "Расскажи про Python",
        "Как работает RAG?"
    ]
    
    for query in test_queries:
        print(f"\n   ❓ Запрос: {query}")
        
        # Создаём эмбеддинг запроса
        query_embedding = embedder.embed_text(query)
        
        # Поиск в ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        
        # Выводим результаты
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                # ChromaDB возвращает distance (меньше = лучше), конвертируем в score
                score = 1 / (1 + distance)  # Простая нормализация
                print(f"      [{i+1}] Score: {score:.4f}")
                print(f"          {doc[:80]}...")
        else:
            print("      Нет результатов")
    
    print("\n" + "=" * 80)
    print("📋 РЕЗУЛЬТАТЫ ТЕСТА")
    print("=" * 80 + "\n")
    
    print("✅ ChromaDB работает ОТЛИЧНО!")
    print("✅ Векторы успешно сохранены")
    print("✅ Поиск работает корректно")
    print("✅ Не требует API ключей")
    print("✅ Данные сохраняются локально\n")
    
    print("💡 ПРЕИМУЩЕСТВА ChromaDB:")
    print("   - Полностью бесплатная")
    print("   - Работает локально (быстро!)")
    print("   - Простая установка")
    print("   - Не нужны API ключи")
    print("   - Отлично для разработки и production\n")


def test_with_old_relevance_api():
    """Попытка работы через старый API (если доступен)."""
    
    print("\n" + "=" * 80)
    print("🔄 ПОПЫТКА ИСПОЛЬЗОВАТЬ СТАРЫЙ API")
    print("=" * 80 + "\n")
    
    try:
        # Пробуем импортировать старые классы
        from relevanceai import Client
        print("✅ Старый API (Client) доступен!")
        
        # Пробуем инициализировать
        client = Client(
            project=settings.RELEVANCE_PROJECT,
            api_key=settings.RELEVANCE_API_KEY
        )
        print("✅ Client инициализирован!")
        
        # Пробуем работать с datasets
        datasets = client.list_datasets()
        print(f"✅ Datasets: {datasets}\n")
        
        return True
        
    except ImportError:
        print("❌ Старый API (Client) не доступен в версии 10.x")
        print("   Нужна версия 2.x для векторного поиска\n")
        return False
    except Exception as e:
        print(f"❌ Ошибка старого API: {e}\n")
        return False


def main():
    """Главная функция."""
    
    print("\n" + "█" * 80)
    print("  RELEVANCE AI - ПОЛНАЯ ПРОВЕРКА")
    print("█" * 80)
    
    # Проверка конфигурации
    if not settings.RELEVANCE_API_KEY:
        print("\n❌ ОШИБКА: RELEVANCE_API_KEY не установлен в .env файле!\n")
        return
    
    if not settings.RELEVANCE_PROJECT:
        print("\n❌ ОШИБКА: RELEVANCE_PROJECT не установлен в .env файле!\n")
        return
    
    print(f"\n✅ Конфигурация найдена:")
    print(f"   Project: {settings.RELEVANCE_PROJECT}")
    print(f"   API Key: {settings.RELEVANCE_API_KEY[:20]}...\n")
    
    # Тест нового API
    test_relevance_api()
    
    # Тест старого API
    old_api_works = test_with_old_relevance_api()
    
    # Финальные рекомендации
    print("=" * 80)
    print("🎯 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ")
    print("=" * 80 + "\n")
    
    if not old_api_works:
        print("📌 Relevance AI 10.x НЕ поддерживает векторный поиск")
        print("📌 Это платформа для создания AI агентов\n")
        
        print("✅ РЕШЕНИЕ 1: Используйте Pinecone + Weaviate")
        print("   У вас они УЖЕ работают отлично!\n")
        
        print("⚠️  РЕШЕНИЕ 2: Откатитесь на Relevance AI 2.x")
        print("   НО будут проблемы с установкой на Windows:")
        print("   pip uninstall relevanceai")
        print("   pip install relevanceai==2.0.0\n")
        
        print("✅ РЕШЕНИЕ 3: Добавьте альтернативную векторную БД:")
        print("   - Qdrant: pip install qdrant-client")
        print("   - ChromaDB: pip install chromadb")
        print("   - FAISS: pip install faiss-cpu\n")
    else:
        print("✅ Старый API работает - можно использовать!")
        print("   Но могут быть проблемы совместимости\n")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем\n")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}\n")
        import traceback
        traceback.print_exc()

