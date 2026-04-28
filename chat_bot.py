import chromadb
import requests
from sentence_transformers import SentenceTransformer

# ==============================
# INIT
# ==============================
embedder = SentenceTransformer('BAAI/bge-m3')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="docs")

# LM Studio API
LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "google/gemma-4-e4b"

# ==============================
# SEARCH CHROMA
# ==============================
def search_chroma(query):
    query_emb = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )

    return results["documents"][0]


# ==============================
# CALL LM STUDIO
# ==============================
def call_llm(prompt):

    response = requests.post(
        LM_API_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
    )

    result = response.json()

    return result["choices"][0]["message"]["content"]


# ==============================
# GENERATE JAWABAN
# ==============================
def generate_answer(query):

    docs = search_chroma(query)

    context = "\n\n".join(docs)

    prompt = f"""
Anda adalah asisten kesehatan ibu dan anak.

Jawab HANYA dari informasi berikut:
{context}

Pertanyaan:
{query}

Jawab singkat, jelas, tidak mengarang.
"""

    return call_llm(prompt)


# ==============================
# MAIN LOOP
# ==============================
if __name__ == "__main__":
    print("=== CHATBOT KESEHATAN (LM STUDIO) ===")
    print("ketik 'exit' untuk keluar\n")

    while True:
        query = input("Anda: ")

        if query.lower() in ["exit", "quit"]:
            break

        try:
            answer = generate_answer(query)
            print("\nChatbot:")
            print(answer)
        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "-"*50)