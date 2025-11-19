from chunkers import SemanticChunker, RecursiveCharacterTextChunker



def process_empty(data):
    return data



def process_single_dict(row):
    text = row["transcript"]

    chunker = SemanticChunker()  # Создаём экземпляр класса
    chunks = chunker.chunk(text)

    new_chunks = []
    for i in range(len(chunks)):
        new_chunk = {"text": chunks[i]}
        keys = list(row.keys())
        keys.remove("transcript")
        for key in keys:
            new_chunk[key] = row[key]
        new_chunks.append(new_chunk)

    return new_chunks


def process_data(data):
    chunks = []
    for row in data:
        chunks.extend(process_single_dict(row))
    return chunks

def preprocess_query_simple(rows):
    return [rows[i]["query"] for i in range(len(rows))]

def preprocess_query123(query):
    return query[0]["categories"][0]["questions"]

def main():
    chunks = process_data([{"transcript": "Today we discussed the new marketing strategy for Q1. The team agreed to focus on short-form video content and improve customer acquisition funnels.", "summary": "Meeting about Q1 marketing strategy and next steps.", "date": "2025-01-17", "attendants": ["Alice Johnson", "Mark Rivera", "Diana Petrova", "Samuel Kim"], "links": ["https://docs.example.com/marketing-plan", "https://drive.example.com/file/strategy-overview"]}])
    print(chunks)


if __name__ == "__main__":
    pass