import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

OLLAMA_URL = "http://localhost:11434"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "faq"

def chunk_text(text, size=400):
    chunks = []
    words = text.replace('→', ' ').split()
    current = []
    current_size = 0

    for word in words:
        current.append(word)
        current_size += len(word) + 1
        if current_size >= size:
            chunks.append(' '.join(current))
            current = []
            current_size = 0

    if current:
        chunks.append(' '.join(current))

    return chunks

def get_embedding(text):
    response = requests.post(f"{OLLAMA_URL}/api/embeddings",
                            json={"model": "embeddinggemma:latest", "prompt": text})
    return response.json()['embedding']

with open('text_pdf.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Chunking text...")
chunks = chunk_text(text)
print(f"Created {len(chunks)} chunks")

print("Getting embeddings...")
embeddings = []
for i, chunk in enumerate(chunks):
    print(f"{i+1}/{len(chunks)}", end='\r')
    embeddings.append(get_embedding(chunk))

print("\nStoring in Qdrant...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
)

points = [
    PointStruct(id=i, vector=embeddings[i], payload={"text": chunks[i]})
    for i in range(len(chunks))
]

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Done! Stored {len(points)} vectors")
