from chunkers import SemanticChunker, RecursiveCharacterTextChunker


def process_data(data):
    chunks = []
    for row in data:
        chunks.extend(process_single_dict(row))
    return chunks




def process_single_dict(row):
    links = row["links"]
    attendants = row["attendants"]
    date = row["date"]
    summary = row["summary"]
    text = row["text"]

    chunker = SemanticChunker()  # Создаём экземпляр класса
    chunks = chunker.chunk(text)

    new_chunks = []
    for i in range(len(chunks)):
        # chunks[i] = f"Transcription:\n{chunks[i]}\n\nSummary:\n{summary}\n\nAttendants:\n{attendants}"
        new_chunk = {"text": chunks[i]}
        keys = list(row.keys())
        keys.remove("text")
        for key in keys:
            new_chunk[key] = row[key]
        new_chunks.append(new_chunk)


    return new_chunks


def main():
    chunks = process_data([{"text": "Today we discussed the new marketing strategy for Q1. The team agreed to focus on short-form video content and improve customer acquisition funnels.", "summary": "Meeting about Q1 marketing strategy and next steps.", "date": "2025-01-17", "attendants": ["Alice Johnson", "Mark Rivera", "Diana Petrova", "Samuel Kim"], "links": ["https://docs.example.com/marketing-plan", "https://drive.example.com/file/strategy-overview"]}])
    print(chunks)


if __name__ == "__main__":
    main()