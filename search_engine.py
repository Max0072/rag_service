from dotenv import load_dotenv
import os
import torch
import chunkers
import faiss
from prompt_builders import BasicPromptBuilder
from LLMs import LLMopenAI
from sentence_transformers import SentenceTransformer
import numpy as np

# safety measure for MacOS cpu
def faiss_threads_to_one():
    os.environ["OMP_NUM_THREADS"] = "1"
    faiss.omp_set_num_threads(1)




# search engine class
class SearchEngine:
    def __init__(self, embed_model, index):
        self.embed_model = embed_model
        self.index = index
        self.id_to_chunk = {}

        faiss_threads_to_one()

# ------------ embedding process ------------------- done 100%

    # Get embedding of a single text sample
    def embed_normalized(self, query):
        self.embed_model.eval()
        embedding = self.embed_model.encode(query, convert_to_numpy=True)
        norm = np.linalg.norm(embedding)
        normalized_embedding = embedding / norm if norm != 0.0 else embedding
        return normalized_embedding.astype(np.float32)

# ------------ working with index and corpus ------------------- done 100%

    # Checks if the index is empty (If it's empty, no search can be performed)
    def index_is_empty(self) -> bool:
        return self.index.ntotal == 0

    # Delete all data and reset the index
    def delete_all_data(self):
        self.id_to_chunk.clear()
        self.index.reset()

    # corpus format [{"text": str, "data": str}, ... ]
    def upload_corpus(self, corpus):
        print("Adding corpus to index...")
        embeddings = []
        ids = []
        for sample in corpus:
            internal_id = len(self.id_to_chunk)
            self.id_to_chunk[internal_id] = sample

            embeddings.append(self.embed_normalized(sample["text"]+sample["meta-data"])) # get embedding of the text part
            ids.append(internal_id)                                 # add current id to id list

        self.index.add_with_ids(np.array(embeddings).astype(np.float32), np.array(ids, dtype=np.int64))
        print("Index built successfully")
        return 0

    # ------------- search utils part ------------------------

    # find top-k similar vector ids and similarity scores
    def _find_similar_from_vector(self, vec, k=10):
        vec = vec.reshape(1, -1).astype(np.float32)
        scores, ids = self.index.search(vec, k)
        return {"ids": ids[0].tolist(), "scores": scores[0].tolist()}

    # --------------- search part -------------------------------------

    # search by a single query
    def similarity_search_single(self, query, k=10):
        embedding = self.embed_normalized(query)
        id_score_dict = self._find_similar_from_vector(embedding, k)
        return id_score_dict                # returns {"ids": ids, "scores": scores}

    # search by list of queries
    def similarity_search_batch(self, queries, k=10):
        print("Starting batch search...")
        embeddings = []
        for query in queries:
            embeddings.append(self.embed_normalized(query))
        id_score_list = []
        for q_emb in embeddings:
            id_score_dict = self._find_similar_from_vector(q_emb, k)
            id_score_list.append(id_score_dict)
        print("Batch search finished successfully")
        return id_score_list                # returns [{"ids": ids, "scores": scores}, ... ]


# ------------- Getting texts from ids ---------------------

    # return a list of chunks
    def get_chunks_by_ids(self, ids):
        chunks = [self.id_to_chunk[_id] for _id in ids]
        return chunks

# -------------------- finish ------------------------------




# ------------ upload raw text with metadata ----------
    # {"text": text, "summary": summary, "date": date, "attendants": [attendants, ...], "link_transcript": , "link_video": ]}




# -----------------  Pipeline --------------------------
#
# Get big text
# 1) Chunking
# 2) To index
# 3) Query search, k
#    At this point we have query and relevant docs with meta-data that we can send to LLM
# 4) Prompt Builder
# 5) ask LLM
#
#
# -----------------------------------------------------

def build_rag():
    # chunker = chunkers.SemanticChunker()
    # chunker = chunkers.RecursiveCharacterTextChunker()




    return search_engine



def main():
    load_dotenv()
    rag = build_rag()










if __name__ == "__main__":


    main()

