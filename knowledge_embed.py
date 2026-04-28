import json
import chromadb
import hashlib
import uuid
from sentence_transformers import SentenceTransformer

# ==============================
# INIT
# ==============================
embedder = SentenceTransformer('BAAI/bge-m3')

#  WAJIB pakai ini
client = chromadb.PersistentClient(path="./chroma_db")

#  HAPUS COLLECTION LAMA (OPTIONAL)
try:
    client.delete_collection("docs")
    print("Collection lama dihapus")
except:
    pass

collection = client.get_or_create_collection(name="docs")

# ==============================
# FILE JSON
# ==============================
files = [
    "hasil_chunking/kia_chunks_refined.json",
    "hasil_chunking/gizi_chunks.json",
    "hasil_chunking/antenatal_chunks_refined.json",
    "hasil_chunking/imunisasi_chunks_refined.json"
]

# ==============================
# LOOP FILE
# ==============================
for file_path in files:
    print(f"\nMemproses file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Gagal baca file {file_path}: {e}")
        continue

    documents = []
    embeddings = []
    ids = []
    metadatas = []

    seen_ids = set()

    for i, item in enumerate(data):
        try:
            # ==============================
            # GENERATE ID AMAN
            # ==============================
            chunk_id_raw = item.get("chunk_id")

            if chunk_id_raw:
                chunk_id = str(chunk_id_raw)

            elif item.get("content"):
                raw_text = item.get("content")[:200]
                chunk_id = hashlib.md5(raw_text.encode()).hexdigest()

            else:
                chunk_id = str(uuid.uuid4())

            if chunk_id in seen_ids:
                print(f"⚠️ duplicate: {chunk_id}, skip")
                continue

            seen_ids.add(chunk_id)

            # ==============================
            # TEXT EMBEDDING
            # ==============================
            text = f"""
{item.get('title', '')}
Kategori: {item.get('category', '')}
Sub: {item.get('sub_category', '')}
Isi: {item.get('content', '')}
"""

            emb = embedder.encode(text).tolist()

            documents.append(text)
            embeddings.append(emb)
            ids.append(chunk_id)

            # ==============================
            # METADATA
            # ==============================
            metadatas.append({
                "title": item.get("title"),
                "category": item.get("category"),
                "sub_category": item.get("sub_category"),
                "type": item.get("type"),
                "keywords": ", ".join(item.get("keywords", [])),
                "priority": item.get("priority"),
                "page": item.get("page"),
                "source": item.get("source")
            })

            print(f"{chunk_id} siap")

        except Exception as e:
            print(f"Error di item ke-{i}: {e}")

    # ==============================
    # INSERT KE CHROMA
    # ==============================
    if documents:
        try:
            collection.upsert(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            print(f"✔ {len(documents)} data masuk ke Chroma")
        except Exception as e:
            print(f" Error insert: {e}")

# ==============================
# DONE
# ==============================
print("\n SEMUA DATA MASUK KE CHROMA")