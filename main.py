from search_engine import SearchEngine
import torch
from sentence_transformers import SentenceTransformer
import faiss

from data_processor import process_empty
from utils import get_device, model_emb_dim


# -----------------  Pipeline --------------------------

# Get big text
# 1) Chunking
# {"text": text, "summary": summary, "date": date, "attendants": [attendants, ...], "link_transcript": , "link_video": }
# 2) To index
# 3) Query search,
#    At this point we have query and relevant docs with meta-data that we can send to LLM
# 4) Prompt Builder
# 5) ask LLM
# -----------------------------------------------------

def main():

    # corpus and queries
    corpus = [
        {"text": "Hello, world!", "meta-data": "Lol, very typical sentence"},
        {"text": "Hello, world!", "meta-data": "Basic sentence in python"},
    ]
    query = "Is it a typical thing?"

    # preprocessing
    corpus = process_empty(corpus)

    # search engine init
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(get_device()))
    dim = model_emb_dim(embed_model)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    search_engine = SearchEngine(embed_model, index)

    # forward data through the search engine
    search_engine.upload_corpus(corpus)
    id_score_list = search_engine.similarity_search_single(query, k=10)

    # filter if len rev < k
    ids = [_id for _id in id_score_list["ids"] if _id != -1]
    scores = [id_score_list["scores"][i] for i in range(len(ids))]
    print(ids)
    print(scores)

    # get chunks
    chunks = search_engine.get_chunks_by_ids(ids)
    print(chunks)


if __name__ == "__main__":
    pass


