# ==============================
# 1. IMPORTS
# ==============================
import os
import sys
import json
import csv
import requests
import re
from datetime import datetime
from collections import deque
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

from config import TOP_K, PROJECT_ROOT

# ==============================
# 2. KONFIGURASI
# ==============================
EMBEDDING_MODEL = str(Path(__file__).resolve().parent / "bge-m3")
DB_PATH = "./chroma_db"
COLLECTION_NAME = "docs"
LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL = "google/gemma-4-e2b"
TEMPERATURE = 0.2
MAX_HISTORY = 3
LOG_FILE = "chatbot_logs.jsonl"
MAX_TOKENS = 2048
TIMEOUT_SECONDS = 180
DATASET_EVAL_FILE = PROJECT_ROOT / "dataset_evaluasi.json"
CSV_INPUT_FILE = PROJECT_ROOT / "data_manual.csv"

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

def log_interaction(
    query: str,
    answer: str,
    context: str = "",
    metadata: dict = None,
    contexts: list = None,
    ground_truth: str = "",
):
    """Simpan interaksi ke file log untuk evaluasi."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "contexts": contexts or [],
        "ground_truth": ground_truth,
        "context_preview": context[:200] + "..." if len(context) > 200 else context,
        "metadata": metadata or {}
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8-sig") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Warning] Gagal menyimpan log: {e}")


def detect_intent(query: str) -> dict:
    """Deteksi intent dasar: greeting dan exit."""
    q = query.lower().strip()
    greeting_patterns = [r"\bhalo\b", r"\bhai\b", r"\bhello\b", r"\bhi\b", r"\bassalamualaikum\b"]
    return {
        "is_greeting": any(re.search(pattern, q) for pattern in greeting_patterns),
        "is_exit": q in ["exit", "quit", "keluar", "bye", "selesai", "terima kasih"]
    }


def truncate_context(context: str, max_chars: int = 1500) -> str:
    """Potong konteks jika terlalu panjang untuk LLM."""
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "\n\n[...konten dipotong...]"


# ==============================
# 5. FUNGSI CORE: RETRIEVAL & GENERATION
# ==============================

def search_documents(query: str, n_results: int = 3):
    """Cari dokumen relevan dari ChromaDB menggunakan semantic search."""
    try:
        query_embedding = embedder.encode(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
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


def generate_response(query: str, ground_truth: str = "") -> tuple:
    """
    Generate jawaban menggunakan pipeline RAG.
    Returns: (jawaban, is_truncated)
    """
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
    
    print("Mencari referensi...", end="\r")
    docs_with_meta = search_documents(query, n_results=TOP_K)
    print(" " * 40, end="\r")
    
    if not docs_with_meta:
        fallback = (
            "Maaf, saya belum menemukan informasi spesifik tentang hal tersebut.\n\n"
            "Saran:\n"
            "- Coba gunakan kata kunci yang berbeda\n"
            "- Konsultasikan dengan tenaga kesehatan di Puskemas Kuranji\n"
            "- Untuk keadaan darurat, segera hubungi 119 atau Puskesmas Kuranji"
        )
        log_interaction(
            query,
            fallback,
            context="NO_RESULTS",
            contexts=[],
            ground_truth=ground_truth,
            metadata={"fallback": True},
        )
        return fallback, False
    
    docs = [d[0] for d in docs_with_meta]
    context = "\n\n---\n\n".join(docs)
    
    if len(context) > 1500:
        context = truncate_context(context)
    
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
    
    print("Menyusun jawaban...", end="\r")
    answer, is_truncated = call_llm(prompt)
    print(" " * 40, end="\r")
    
    if is_truncated:
        answer += "\n\n*(Jawaban terpotong. Silakan tanya 'lanjutkan' untuk melanjutkan)*"
    
    conversation_history.append({
        "query": query,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    })
    
    log_interaction(
        query=query,
        answer=answer,
        context=context,
        contexts=docs,
        ground_truth=ground_truth,
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
        with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
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
# 7. MODE KONVERSI CSV KE JSON
# ==============================

def convert_csv_to_json():
    """
    Konversi file CSV manual (data_manual.csv) menjadi dataset_evaluasi.json.
    File CSV harus memiliki 2 kolom: 'question' dan 'ground_truth'.
    """
    print("=" * 60)
    print("MODE KONVERSI CSV KE JSON")
    print("=" * 60)
    
    if not CSV_INPUT_FILE.exists():
        print(f"[ERROR] File {CSV_INPUT_FILE} tidak ditemukan.")
        print(f"   Silakan buat file CSV terlebih dahulu dengan format:")
        print(f"   Kolom 1: question")
        print(f"   Kolom 2: ground_truth")
        print(f"   Contoh isi:")
        print(f"   question,ground_truth")
        print(f"   \"Apa itu ASI?\",\"ASI adalah air susu ibu...\"")
        return
    
    dataset = []
    try:
        with open(CSV_INPUT_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "question" not in row or "ground_truth" not in row:
                    print("[WARNING] File CSV harus memiliki kolom 'question' dan 'ground_truth'")
                    print(f"   Kolom yang ditemukan: {list(row.keys())}")
                    return
                
                if not row["question"].strip() or not row["ground_truth"].strip():
                    continue
                
                dataset.append({
                    "question": row["question"].strip(),
                    "ground_truth": row["ground_truth"].strip()
                })
    except Exception as e:
        print(f"[ERROR] Gagal membaca file CSV: {e}")
        return
    
    if not dataset:
        print("[ERROR] File CSV kosong atau tidak ada data valid.")
        return
    
    try:
        with open(DATASET_EVAL_FILE, "w", encoding="utf-8-sig") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n[OK] Berhasil mengonversi {len(dataset)} pertanyaan.")
        print(f"File tersimpan di: {DATASET_EVAL_FILE}")
        print("\nLangkah selanjutnya:")
        print("   python chat_bot.py eval    -> Uji semua pertanyaan otomatis")
        print("   python chat_bot.py chat    -> Chat interaktif")
        print("=" * 60)
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan file JSON: {e}")


# ==============================
# 8. MODE EVALUASI
# ==============================

def run_evaluation():
    """
    Menjalankan chatbot secara otomatis menggunakan dataset evaluasi 
    untuk mengisi nilai ground_truth pada log.
    """
    print("\n" + "="*60)
    print("MEMULAI MODE EVALUASI RAG")
    print("="*60)

    if not DATASET_EVAL_FILE.exists():
        print(f"[ERROR] File dataset evaluasi tidak ditemukan: {DATASET_EVAL_FILE}")
        print("   Jalankan 'python chat_bot.py convert' terlebih dahulu.")
        return

    try:
        with open(DATASET_EVAL_FILE, "r", encoding="utf-8-sig") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"[ERROR] Gagal memuat dataset: {e}")
        return

    if not dataset:
        print("[ERROR] Dataset evaluasi kosong.")
        return

    print(f"[OK] Dataset berhasil dimuat: {len(dataset)} pertanyaan")
    print(f"Log akan disimpan ke: {LOG_FILE}")
    print("="*60 + "\n")

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print("[INFO] Log lama dihapus untuk evaluasi baru.\n")

    for i, item in enumerate(dataset, 1):
        query = item.get("question", "").strip()
        ground_truth = item.get("ground_truth", "").strip()
        
        if not query:
            continue
        
        print(f"[{i}/{len(dataset)}] Pertanyaan: {query[:70]}...")
        
        intents = detect_intent(query)
        if intents["is_greeting"]:
            print("   [SKIP] Greeting detected")
            continue

        docs_with_meta = search_documents(query, n_results=TOP_K)
        
        if not docs_with_meta:
            answer = "Maaf, tidak ditemukan informasi."
            docs = []
            context = "NO_RESULTS"
        else:
            docs = [d[0] for d in docs_with_meta]
            context = "\n\n---\n\n".join(docs)
            if len(context) > 1500:
                context = truncate_context(context)
                
            prompt = f"""Anda adalah asisten kesehatan ibu dan anak yang terpercaya.

INFORMASI SUMBER (gunakan HANYA ini untuk menjawab):
{context}

PERTANYAAN PENGGUNA:
{query}

INSTRUKSI:
1. Jawab berdasarkan informasi sumber di atas.
2. Jika informasi tidak cukup, katakan dengan jujur.
3. Gunakan bahasa Indonesia yang jelas.

JAWABAN:"""
            answer, _ = call_llm(prompt)

        log_interaction(
            query=query,
            answer=answer,
            context=context,
            contexts=docs,
            ground_truth=ground_truth,
            metadata={
                "doc_count": len(docs),
                "evaluation_mode": True,
                "source_title": item.get("source_title", ""),
                "source_page": item.get("source_page", "")
            }
        )
        print(f"   [OK] Jawaban & ground_truth tersimpan.\n")

    print("="*60)
    print(f"[SELESAI] {len(dataset)} pertanyaan telah diuji.")
    print(f"File log: {LOG_FILE}")
    print(f"Silakan jalankan RAGAS untuk menghitung 4 metrik evaluasi.")
    print("="*60 + "\n")


# ==============================
# 9. MAIN LOOP
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
    print("-" * 60)
    print("\nMode lain (jalankan dari terminal):")
    print("   python chat_bot.py convert  -> Konversi CSV ke JSON")
    print("   python chat_bot.py eval     -> Evaluasi otomatis")
    print("-" * 60 + "\n")
    
    while True:
        try:
            query = input("Anda: ").strip()
            
            if not query:
                continue
            
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


# ==============================
# 10. ENTRY POINT DENGAN ARGUMEN
# ==============================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "convert":
            convert_csv_to_json()
            
        elif mode == "eval":
            run_evaluation()
            
        elif mode == "chat":
            main()
            
        else:
            print("[ERROR] Mode tidak dikenali.")
            print("\nCara penggunaan:")
            print("   python chat_bot.py convert  -> Konversi CSV manual ke JSON")
            print("   python chat_bot.py eval     -> Evaluasi otomatis semua pertanyaan")
            print("   python chat_bot.py chat     -> Chat interaktif (default)")
    else:
        main()