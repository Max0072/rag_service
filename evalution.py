import numpy as np
from datasets_for_evaluation import load_nq_dataset_sample
from search_engine import build_rag
import pickle

def save_var(my_var, path):
    with open(path, "wb") as f:
        pickle.dump(my_var, f)

def load_var(path):
    with open(path, "rb") as f:
        my_var = pickle.load(f)
        return my_var

class Evaluation:
    def __init__(self):
        self.id_to_doc_id = {}
        self.doc_id_to_id = {}
        self.doc_id_to_corpus = {}

        self.id_to_q_id = {}
        self.q_id_to_id = {}
        self.q_id_to_query = {}

    def prepare_corpus(self, corpus):
        docs = []
        for doc_id, doc in corpus.items():
            internal_id = len(self.id_to_doc_id)
            self.id_to_doc_id[internal_id] = doc_id
            self.doc_id_to_id[doc_id] = internal_id
            self.doc_id_to_corpus[doc_id] = doc
            docs.append(doc)
        return docs

    def prepare_queries(self, queries):
        qs = []
        for q_id, q in queries.items():
            internal_id = len(self.id_to_q_id)
            self.id_to_q_id[internal_id] = q_id
            self.q_id_to_id[q_id] = internal_id
            self.q_id_to_query[q_id] = q
            qs.append(q)
        return qs



    def evaluate_similarity_search(self):
        corpus, queries, qrels = load_nq_dataset_sample()
        prepared_corpus = self.prepare_corpus(corpus)
        prepared_queries = self.prepare_queries(queries)

        rag = build_rag()

        rag.upload_corpus(prepared_corpus)
        print(f"Index is empty: {rag.index_is_empty()} (False means not empty lol)")
        k = 10
        id_score_list = rag.similarity_search_batch(prepared_queries, k=k)

        save_var(qrels, "qrels.pkl")
        save_var(id_score_list, "./id_score_list.pkl")
        save_var(self.id_to_q_id, "./id_to_q_id.pkl")
        save_var(self.id_to_doc_id, "./id_to_doc_id.pkl")


        qrels = load_var("qrels.pkl")
        id_score_list = load_var("./id_score_list.pkl")
        self.id_to_q_id = load_var("./id_to_q_id.pkl")
        self.id_to_doc_id = load_var("./id_to_doc_id.pkl")


        print("Start evaluation...")
        recall_k_list = []
        mrr_k_list = []
        ndcg_list = []
        for i in range(len(id_score_list)):
            ids = id_score_list[i]["ids"]
            doc_ids = [self.id_to_doc_id[_id] for _id in ids]
            query_id = self.id_to_q_id[i]
            relevancy_list = [1 if doc_id in qrels[query_id].keys() else 0 for doc_id in doc_ids]

            num_total_relevant_docs = len(qrels[query_id])
            num_relevant_docs = relevancy_list.count(1)

            relevant_ranks = [i + 1 for i in range(len(relevancy_list)) if relevancy_list[i] == 1]
            first_relevant_rank = relevancy_list.index(1) + 1 if 1 in relevancy_list else 0


            recall_k = num_relevant_docs / num_total_relevant_docs
            mrr_k = 1/first_relevant_rank if first_relevant_rank > 0 else 0

            dcg_k = sum([1/np.log2(i + 1) for i in relevant_ranks])
            idcg_k = sum([1/np.log2(i + 1) for i in range(1, min(num_total_relevant_docs, k)+1)])
            ndcg_k = dcg_k / idcg_k

            recall_k_list.append(recall_k)
            mrr_k_list.append(mrr_k)
            ndcg_list.append(ndcg_k)

        batch_recall_k = np.mean(recall_k_list)
        batch_mrr_k = np.mean(mrr_k_list)
        batch_ndcg_k = np.mean(ndcg_list)

        return batch_recall_k, batch_mrr_k, batch_ndcg_k


def main():

    evaluator = Evaluation()
    recall_k, mrr_k, ndcg_k = evaluator.evaluate_similarity_search()

    print(f"Recall: {recall_k}")
    print(f"MRR: {mrr_k}")
    print(f"NDCG: {ndcg_k}")

if __name__ == "__main__":
    main()


# With semantic chanking
# Recall: 0.9402761684047894
# MRR: 0.8814424350273135
# NDCG: 0.8883600267658702