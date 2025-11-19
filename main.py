from search_engine import SearchEngine
import torch
from sentence_transformers import SentenceTransformer
import faiss

from utils import get_device


def main():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(get_device()))
    index = faiss.IndexIDMap(faiss.IndexFlatIP(384))

    search_engine = SearchEngine(embed_model, index)

    corpus = [
        {"text": "Hello, world!", "meta-data": "Lol, very typical sentence"},
        {"text": "Hello, world!", "meta-data": "Basic sentence in python"},
    ]

    query = "Is it a typical thing?"

    search_engine.upload_corpus(corpus)
    id_score_list = search_engine.similarity_search_single(query, k=2)
    ids = id_score_list["ids"]
    scores = id_score_list["scores"]
    print(ids)
    print(scores)
    relevant_chunks = search_engine.get_chunks_by_ids(ids)
    print(relevant_chunks)


if __name__ == "__main__":
    main()


