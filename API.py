from typing import List, Any, Optional, Dict
from fastapi import FastAPI, UploadFile, File, Request, Body

from utils import parse_json_or_jsonl, get_device
from data_processor import process_data, preprocess_query123
from search_engine import SearchEngine

from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI(title="VectorSearchAPI")


embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(get_device()))
index = faiss.IndexIDMap(faiss.IndexFlatIP(384))

search_engine = SearchEngine(embed_model, index)



# jsons:
# {"transcript":, "summary":, "date":, "attendants":, "links":}
@app.post("/upload_json")
async def upload_json(request: Request, file: Optional[UploadFile] = File(None)):
    if file is not None:
        raw = await file.read()
        rows_parsed = parse_json_or_jsonl(raw)
    else:
        # Читаем сырое тело запроса
        body = await request.body()
        if not body:
            return {"error": "No data provided"}
        rows_parsed = parse_json_or_jsonl(body)
    print(type(rows_parsed))
    print(rows_parsed)

    # Process_data
    processed_chunks = process_data(rows_parsed)
    # Upload corpus to the search engine
    search_engine.upload_corpus(processed_chunks)
    return {"loaded": len(rows_parsed)}


@app.delete("/delete_all")
def delete_all():
    search_engine.delete_all_data()
    return {"status": "all data deleted"}


@app.post("/search")
def search(query: dict):
    if search_engine.index_is_empty():
        return {"detail": "Index is empty. Upload data first."}
    text = query["query"]

    # similarity search by the single query
    id_score_list = search_engine.similarity_search_single(text, k=10)
    # returns {"ids": ids, "scores": scores}

    ids = [_id for _id in id_score_list["ids"] if _id != -1]
    scores = [id_score_list["scores"][i] for i in range(len(ids))]
    print(ids)
    print(scores)

    chunks = search_engine.get_chunks_by_ids(ids)
    return {"chunks": chunks}


@app.post("/search_batch")
async def search_batch(request: Request, file: UploadFile | None = File(None)):
    if search_engine.index_is_empty():
        return {"detail": "Index is empty. Upload data first."}
    if file:
        raw = await file.read()
        rows = parse_json_or_jsonl(raw)
    else:
        rows = await request.json()

    queries = preprocess_query123(rows)

    # similarity search by the multiple queries
    id_score_lists = search_engine.similarity_search_batch(queries, k=10)
    # returns [{"ids": ids, "scores": scores}, ... ]

    # filter ids and scores if num rel chunks < k
    ids = []
    scores = []
    for entry in id_score_lists:
        filtered_ids = [_id for _id in entry["ids"] if _id != -1]
        filtered_scores = [entry["scores"][i] for i in range(len(filtered_ids))]
        ids.append(filtered_ids)
        scores.append(filtered_scores)
    print(ids)
    print(scores)

    chunks = search_engine.get_chunks_by_ids(ids)

    return {"chunks": chunks}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API:app", host="0.0.0.0", port=8000, reload=True)
