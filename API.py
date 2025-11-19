import json
from typing import List, Any, Optional, Dict
from fastapi import FastAPI, UploadFile, File, Request, Body
from sentence_transformers import SentenceTransformer
from search_engine import build_rag
from data_processor import process_data
# This api allows uploading JSON/JSONL data and perform similarity search.
# Read the README_API.md for details.

app = FastAPI(title="VectorSearchAPI")

search_engine = build_rag()


# function to parse JSON or JSONL from bytes
def parse_json_or_jsonl(raw: bytes, encoding: str = 'utf-8') -> List[Any]:
    text = raw.decode(encoding).strip()
    if text.startswith('['):
        data = json.loads(text)
        if isinstance(data, list):
            return data
        else:
            raise ValueError("JSON content is not a list")
    items = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items



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

    processed_chunks = process_data(rows_parsed)
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
    text = query.get("query")
    ids_score_list = search_engine.similarity_search_single(text)
    ids = ids_score_list["ids"]
    chunks = search_engine.get_chunks_by_ids(ids)
    return chunks


# @app.post("/search_batch")
# async def search_batch(request: Request, file: UploadFile | None = File(None)):
#     if search_engine.index_is_empty():
#         return {"detail": "Index is empty. Upload data first."}
#     if file:
#         raw = await file.read()
#         rows = parse_json_or_jsonl(raw)
#     else:
#         rows = await request.json()
#
#     results = search_engine.find_similar_from_text_batch(rows)
#     return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API:app", host="0.0.0.0", port=8000, reload=True)