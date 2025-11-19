from beir import util
from beir.datasets.data_loader import GenericDataLoader


def download_nq_dataset():
    # Скачиваем и распаковываем датасет
    dataset_name = "nq"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, "datasets")
    return data_path

def load_nq_dataset(data_path="datasets/nq"):
    # Загружаем данные
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

def load_nq_dataset_sample(data_path="datasets/nq", max_docs=10000):
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # Берём только документы, которые есть в qrels (релевантные)
    relevant_doc_ids = set()
    for query_docs in qrels.values():
        relevant_doc_ids.update(query_docs.keys())

    # Фильтруем corpus
    filtered_corpus = {doc_id: corpus[doc_id] for doc_id in relevant_doc_ids if doc_id in corpus}

    return filtered_corpus, queries, qrels

def main():
    data_path = "datasets/nq"
    corpus, queries, qrels = load_nq_dataset(data_path)

    print(f"Corpus size: {len(corpus)}")
    print(f"Queries size: {len(queries)}")
    print(f"Qrels size: {len(qrels)}")

    # # Пример использования
    # print("\nПример query:")
    # query_id = list(queries.keys())[0]
    # print(f"ID: {query_id}")
    # print(f"Text: {queries[query_id]}")

    # print("\nПример документа:")
    # doc_id = list(corpus.keys())[0]
    # print(f"ID: {doc_id}")
    # print(f"Title: {corpus[doc_id].get('title', 'N/A')}")
    # print(f"Text: {corpus[doc_id]['text'][:200]}...")

    # Пример использования
    print("\nПример qrel:")
    qrel_id = list(qrels.keys())[0]
    print(f"ID: {qrel_id}")
    print(f"Text: {qrels[qrel_id]}")

if __name__ == "__main__":
    main()

# Структура данных:
#
#   1. corpus - Корпус документов
#
#   corpus = {
#       "doc_id_1": {
#           "title": "Заголовок статьи",
#           "text": "Полный текст документа..."
#       },
#       "doc_id_2": {...},
#       ...
#   }
#   - Это база знаний (Wikipedia статьи)
#   - Каждый документ имеет ID, заголовок и текст
#   - Размер: ~2.7 млн документов
#
#   2. queries - Запросы пользователей
#
#   queries = {
#       "query_id_1": "когда был основан Google?",
#       "query_id_2": "что такое машинное обучение",
#       ...
#   }
#   - Реальные вопросы пользователей
#   - Каждый запрос имеет уникальный ID
#   - Размер: ~3,452 запроса в test split
#
#   3. qrels - Relevance judgments (метки релевантности)
#
#   qrels = {
#       "query_id_1": {
#           "doc_id_5": 1,    # этот документ релевантен запросу
#           "doc_id_42": 1,
#       },
#       "query_id_2": {
#           "doc_id_123": 1,
#       },
#       ...
#   }
#   - Показывает какие документы релевантны каким запросам
#   - 1 = релевантен, 0 = не релевантен
#   - Используется для оценки качества вашей RAG системы
