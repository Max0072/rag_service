from search_engine import SearchEngine
import torch
from sentence_transformers import SentenceTransformer
import faiss


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")










def main():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(get_device()))
    index = faiss.IndexIDMap(faiss.IndexFlatIP(384))


    search_engine = SearchEngine(embed_model, index)









if __name__ == "__main__":
    main()


