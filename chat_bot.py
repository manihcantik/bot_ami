# ==============================
# 1. IMPORTS
# ==============================
import os
import json
import requests
from datetime import datetime
from collections import deque
from sentence_transformers import SentenceTransformer
import chromadb

# ==============================
# 2. KONFIGURASI
# ==============================
EMBEDDING_MODEL = "BAAI/bge-m3"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "docs"
LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL = "google/gemma-4-e2b"
TEMPERATURE = 0.2          # untuk meminimalkan hallucination dan memastikan jawaban bersifat faktual
MAX_HISTORY = 3            # Membatasi memori percakapan terakhir agar konteks tetap relevan tanpa membebani kapasitas token model lokal
LOG_FILE = "chatbot_logs.jsonl"
MAX_TOKENS = 2048
TIMEOUT_SECONDS = 180

# ==============================
# 3. INISIALISASI GLOBAL
# ==============================
print("[1/3] Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("[2/3] Menghubungkan ke ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

print("[3/3] Menyiapkan memori percakapan...")
conversation_history = deque(maxlen=MAX_HISTORY)

print("\nChatbot siap digunakan!\n")

# ==============================
# 4. FUNGSI HELPER
# ==============================

def log_interaction(query: str, answer: str, context: str = "", metadata: dict = None):
    """Simpan interaksi ke file log untuk evaluasi."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "context_preview": context[:200] + "..." if len(context) > 200 else context,
        "metadata": metadata or {}
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Warning] Gagal menyimpan log: {e}")


def detect_intent(query: str) -> dict:
    """Deteksi intent dasar: greeting dan exit."""
    q = query.lower().strip()
    return {
        "is_greeting": any(kw in q for kw in ["halo", "hai", "hello", "hi", "assalamualaikum"]),
        "is_exit": q in ["exit", "quit", "keluar", "bye", "selesai", "terima kasih"]
    }


def truncate_context(context: str, max_chars: int = 1500) -> str:
    """Potong konteks jika terlalu panjang untuk LLM."""
    if len(context) <= max_chars:
        return context
    # Strategi: ambil bagian awal saja (lebih sederhana)
    return context[:max_chars] + "\n\n[...konten dipotong...]"


# ==============================
# 5. FUNGSI CORE: RETRIEVAL & GENERATION
# ==============================

def search_documents(query: str, n_results: int = 3):
    """Cari dokumen relevan dari ChromaDB menggunakan semantic search."""
    try:
        # Encode query menjadi vektor
        query_embedding = embedder.encode(query).tolist()
        
        # Query ke ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Ekstrak dokumen dan metadata
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        return list(zip(docs, metadatas)) if docs else []
        
    except Exception as e:
        print(f"[Error Retrieval] {e}")
        return []


def call_llm(prompt: str, max_tokens: int = None, timeout: int = None) -> tuple:
    """
    Kirim prompt ke LM Studio API.
    Returns: (jawaban, is_truncated)
    """
    max_tokens = max_tokens or MAX_TOKENS
    timeout = timeout or TIMEOUT_SECONDS
    
    try:
        response = requests.post(
            LM_API_URL,
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TEMPERATURE,
                "max_tokens": max_tokens
            },
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # Deteksi apakah jawaban terpotong
        finish_reason = result["choices"][0].get("finish_reason", "")
        is_truncated = (
            finish_reason == "length" or 
            (content and len(content) > 100 and content.rstrip()[-1] not in ".!?'\"")
        )
        
        return content, is_truncated
        
    except requests.exceptions.Timeout:
        return "[Error] Timeout: Coba pertanyaan yang lebih singkat.", False
    except requests.exceptions.ConnectionError:
        return "[Error] Pastikan LM Studio berjalan di http://127.0.0.1:1234", False
    except Exception as e:
        return f"[Error] {type(e).__name__}: {e}", False


def generate_response(query: str) -> tuple:
    """
    Generate jawaban menggunakan pipeline RAG.
    Returns: (jawaban, is_truncated)
    """
    # Handle greeting langsung tanpa panggil LLM
    intents = detect_intent(query)
    if intents["is_greeting"]:
       return (
            "Halo! Saya adalah asisten edukasi kesehatan ibu dan anak.\n"
            "Saya siap membantu memberikan informasi seputar:\n\n"
            "- Kehamilan, persiapan persalinan, dan masa nifas\n"
            "- Tumbuh kembang bayi dan anak\n"
            "- Imunisasi dan kesehatan anak\n"
            "- Gizi ibu hamil, menyusui, serta nutrisi anak\n"
            "- Keluhan umum pada ibu dan anak\n\n"
            "Silakan ajukan pertanyaan yang ingin Anda ketahui.",
            False
        )
    
    # Stage 1: Retrieval - Cari dokumen relevan
    print("Mencari referensi...", end="\r")
    docs_with_meta = search_documents(query, n_results=3)
    print(" " * 40, end="\r")  # Clear line
    
    # Handle jika tidak ada dokumen ditemukan
    if not docs_with_meta:
        fallback = (
            "Maaf, saya belum menemukan informasi spesifik tentang hal tersebut.\n\n"
            "Saran:\n"
            "- Coba gunakan kata kunci yang berbeda\n"
            "- Konsultasikan dengan tenaga kesehatan di Puskemas Kuranji\n"
            "- Untuk keadaan darurat, segera hubungi 119 atau Puskesmas Kuranji"
        )
        log_interaction(query, fallback, context="NO_RESULTS", metadata={"fallback": True})
        return fallback, False
    
    # Bangun konteks dari dokumen yang ditemukan
    docs = [d[0] for d in docs_with_meta]
    context = "\n\n---\n\n".join(docs)
    
    # Potong konteks jika terlalu panjang
    if len(context) > 1500:
        context = truncate_context(context)
    
    # Stage 2: Generation - Buat prompt dengan grounding constraint
    prompt = f"""Anda adalah asisten kesehatan ibu dan anak yang terpercaya.

INFORMASI SUMBER (gunakan HANYA ini untuk menjawab):
{context}

PERTANYAAN PENGGUNA:
{query}

INSTRUKSI:
1. Jawab berdasarkan informasi sumber di atas
2. Jika informasi tidak cukup, katakan dengan jujur
3. Gunakan bahasa Indonesia yang jelas dan mudah dipahami
4. Untuk kondisi darurat, sarankan untuk segera ke Puskemas Kuranji
5. Gunakan format poin (-) untuk daftar jika diperlukan

JAWABAN:"""
    
    # Panggil LLM untuk generate jawaban
    print("Menyusun jawaban...", end="\r")
    answer, is_truncated = call_llm(prompt)
    print(" " * 40, end="\r")  # Clear line
    
    # Tambahkan indikator jika jawaban terpotong
    if is_truncated:
        answer += "\n\n*(Jawaban terpotong. Silakan tanya 'lanjutkan' untuk melanjutkan)*"
    
    # Simpan ke history untuk konteks percakapan berikutnya
    conversation_history.append({
        "query": query,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    })
    
    # Log untuk evaluasi
    log_interaction(
        query=query,
        answer=answer,
        context=context,
        metadata={
            "doc_count": len(docs),
            "truncated": is_truncated,
            "answer_length": len(answer)
        }
    )
    
    return answer, is_truncated


# ==============================
# 6. UTILITAS
# ==============================

def add_document(doc_id: str, text: str, metadata: dict = None) -> bool:
    """Tambah dokumen baru ke ChromaDB."""
    try:
        embedding = embedder.encode(text).tolist()
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}]
        )
        print(f"OK: Dokumen '{doc_id}' berhasil ditambahkan.")
        return True
    except Exception as e:
        print(f"Gagal menambah dokumen: {e}")
        return False


def show_logs(n: int = 5):
    """Tampilkan n entri log terakhir."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                print("Belum ada log yang tersimpan.")
                return
            
            print(f"\n{n} interaksi terakhir:\n" + "-" * 60)
            for i, line in enumerate(reversed(lines[-n:]), 1):
                entry = json.loads(line.strip())
                print(f"{i}. [{entry['timestamp'][-8:]}] {entry['query'][:50]}...")
                print(f"   -> {entry['answer'][:80]}{'...' if len(entry['answer'])>80 else ''}\n")
    except FileNotFoundError:
        print("File log belum ditemukan.")
    except Exception as e:
        print(f"Error membaca log: {e}")


# ==============================
# 7. MAIN LOOP
# ==============================

def main():
    """Entry point aplikasi chatbot."""
    print("=" * 60)
    print("CHATBOT EDUKASI KESEHATAN IBU DAN ANAK")
    print("=" * 60)
    print("Perintah khusus:")
    print("  /exit   -> Keluar dari chatbot")
    print("  /logs   -> Lihat riwayat interaksi")
    print("  /help   -> Tampilkan panduan ini")
    print("-" * 60 + "\n")
    
    while True:
        try:
            query = input("Anda: ").strip()
            
            if not query:
                continue
            
            # Handle perintah khusus
            if query.lower() in ["/exit", "exit", "keluar", "selesai"]:
                print("\nTerima kasih telah menggunakan chatbot ini!")
                print("Disclaimer: Chatbot ini bukan pengganti konsultasi medis profesional.")
                break
            
            elif query.lower() in ["/logs", "logs", "riwayat"]:
                show_logs(10)
                continue
            
            
            elif query.lower() in ["/help", "help", "?"]:
                print("\nPANDUAN PENGGUNAAN:")
                print("- Ketik pertanyaan kesehatan ibu/anak secara alami")
                print("- Contoh: 'Apa jadwal imunisasi bayi 6 bulan?'")
                print("- Untuk darurat: hubungi 119 atau Puskesmas Kuranji\n")
                continue
            
            # Generate dan tampilkan jawaban
            print("Chatbot: ", end="", flush=True)
            answer, was_truncated = generate_response(query)
            print(answer)
            
            if was_truncated:
                print("\nTips: Ketik 'lanjutkan' untuk melanjutkan jawaban.")
            
            print("\n" + "-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nInterupsi terdeteksi. Keluar dengan aman...")
            break
        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}")
            print("Solusi: Periksa koneksi LM Studio atau restart aplikasi.\n")


if __name__ == "__main__":
    main()