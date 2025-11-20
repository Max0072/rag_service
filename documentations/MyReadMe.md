
# API

```
curl -X POST "https://ragservice-production-f1b3.up.railway.app/upload_json" \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Today we discussed the new marketing strategy for Q1. The team agreed to focus on short-form video content and improve customer acquisition funnels.", "summary": "Meeting about Q1 marketing strategy and next steps.", "date": "2025-01-17", "attendants": ["Alice Johnson", "Mark Rivera", "Diana Petrova", "Samuel Kim"], "links": ["https://docs.example.com/marketing-plan", "https://drive.example.com/file/strategy-overview"], "meta-data": "meta-data"}'
```
```
curl -X POST "https://ragservice-production-f1b3.up.railway.app/upload_json" \
  -H "accept: application/json" \
  -F "file=@./test_data_for_api/test_calls.json"
```

```
curl -X POST "https://ragservice-production-f1b3.up.railway.app/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "where?"}'
```
---------------------------------------

```
curl -X POST "http://0.0.0.0:8000/upload_json" \
  -H "accept: application/json" \
  -F "file=@./test_data_for_api/test_calls.json"
  
```
```
curl -X POST "http://0.0.0.0:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "where?"}'
```




# SearchEngine

### Working with index
index_is_empty
delete_all_data
upload_corpus

### Similarity search

similarity_search_single
similarity_search_batch

### Get actual chunks
get_chunks_by_ids


So we can load the corpus and get top relevant info


