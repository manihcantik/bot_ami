# ==============================
# 1. IMPORTS
# ==============================
import os
import sys
import json
import csv
import requests
import re
import warnings
from datetime import datetime
from collections import deque
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

# ==============================
# 2. KONFIGURASI
# ==============================
TOP_K = 3
PROJECT_ROOT = Path(__file__).resolve().parent

EMBEDDING_MODEL = "BAAI/bge-m3"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "docs"
LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL = "google/gemma-4-e2b"
TEMPERATURE = 0.2
MAX_HISTORY = 3
LOG_FILE = "chatbot_logs.jsonl"
MAX_TOKENS = 4096
TIMEOUT_SECONDS = 180
DATASET_EVAL_FILE = PROJECT_ROOT / "dataset_evaluasi.json"
CSV_INPUT_FILE = PROJECT_ROOT / "data_manual.csv"
RELEVANCE_THRESHOLD = 0.35

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
    clean_q = re.sub(r'[^\w\s]', '', q).strip()
    
    greeting_words = [
        "halo", "hai", "hello", "hi", "assalamualaikum",
        "selamat pagi", "selamat siang", "selamat sore", "selamat malam"
    ]
    is_only_greeting = clean_q in greeting_words
    
    exit_words = ["exit", "quit", "keluar", "bye", "selesai", "terima kasih"]
    is_exit = clean_q in exit_words

    return {
        "is_greeting": is_only_greeting,
        "is_exit": is_exit
    }


def truncate_context(context: str, max_chars: int = 1500) -> str:
    """Potong konteks di akhir kalimat terakhir agar tidak memotong kata."""
    if len(context) <= max_chars:
        return context
    
    cut_text = context[:max_chars]
    last_period = cut_text.rfind('.')
    if last_period > max_chars * 0.8:
        return cut_text[:last_period+1] + "\n\n[...konten dipotong...]"
    
    return cut_text + "\n\n[...konten dipotong...]"


def expand_query(query: str) -> str:
    """Memperluas query pendek dengan sinonim/kata kunci terkait."""
    q = query.lower()
    
    expansions = {
        "hamil": ["kehamilan", "ibu hamil", "trimester", "janin", "gravida"],
        "ciri": ["gejala", "tanda", "keluhan", "simtom"],
        "halangan": ["menstruasi", "haid", "mens", "datang bulan", "period"],
        "nifas": ["postpartum", "setelah melahirkan", "pasca persalinan"],
        "asi": ["air susu ibu", "menyusui", "laktasi", "breastfeeding"],
        "imunisasi": ["vaksin", "vaksinasi", "suntik"],
        "bayi": ["newborn", "neonatus", "balita", "anak"],
        "persalinan": ["melahirkan", "birth", "labor"],
        "gugur": ["miscarriage", "keguguran", "abortus"],
        "kontrasepsi": ["kb", "keluarga berencana", "pil kb"],
        "darah": ["pendarahan", "bleeding", "flek"],
    }
    
    expanded_terms = set()
    words = q.split()
    
    for word in words:
        for key, synonyms in expansions.items():
            if key in word or word in key:
                expanded_terms.update(synonyms)
    
    if expanded_terms:
        synonym_list = list(expanded_terms)[:5]
        return f"{query} {' '.join(synonym_list)}"
    
    return query


def resolve_references(query: str) -> str:
    """
    PERBAIKAN KUNCI: Deteksi kata referensi dan gabungkan dengan konteks sebelumnya.
    Menangani: "hal demikian", "itu", "tersebut", "halangan", dll.
    """
    q = query.lower().strip()
    
    # Kata-kata yang menunjukkan referensi ke percakapan sebelumnya
    reference_words = [
        "hal demikian", "hal itu", "hal tersebut", "itu", "tersebut",
        "tadi", "sebelumnya", "yang itu", "yang tadi", "hal yang sama",
        "gejala tersebut", "gejala itu", "kondisi tersebut", "kondisi itu",
        "penyakit itu", "penyakit tersebut", "masalah itu", "masalah tersebut"
    ]
    
    has_reference = any(ref in q for ref in reference_words)
    
    if has_reference and conversation_history:
        # Ambil konteks dari percakapan sebelumnya
        last_turn = conversation_history[-1]
        last_query = last_turn.get("query", "")
        last_answer = last_turn.get("answer", "")
        
        # Gabungkan query saat ini dengan query dan jawaban sebelumnya
        # Ini memastikan retrieval mengambil dokumen yang relevan dengan topik sebelumnya
        enhanced_query = f"{query} {last_query}"
        
        # Jika ada kata "halangan", tambahkan konteks kehamilan
        if "halangan" in q or "mens" in q or "haid" in q:
            enhanced_query += " kehamilan menstruasi haid"
        
        return enhanced_query
    
    return query


def clean_response(answer: str, previous_answer: str = "") -> str:
    """Potong paksa frasa pembuka yang tidak diinginkan dari JAWABAN APAPUN."""
    if not answer:
        return answer
    
    # DAFTAR LENGKAP frasa pembuka yang HARUS dihapus
    unwanted_patterns = [
        # Frasa "Berdasarkan..."
        r"^[Bb]erdasarkan informasi[^.]*?[:\.]\s*",
        r"^[Bb]erdasarkan teks[^.]*?[:\.]\s*",
        r"^[Bb]erdasarkan sumber[^.]*?[:\.]\s*",
        r"^[Bb]erdasarkan data[^.]*?[:\.]\s*",
        r"^[Bb]erdasarkan referensi[^.]*?[:\.]\s*",
        r"^[Bb]erdasarkan catatan[^.]*?[:\.]\s*",
        r"^[Bb]erdasarkan materi[^.]*?[:\.]\s*",
        r"^[Bb]erdasarkan penjelasan[^.]*?[:\.]\s*",
        
        # Frasa "Dari..."
        r"^[Dd]ari informasi[^.]*?[:\.]\s*",
        r"^[Dd]ari sumber[^.]*?[:\.]\s*",
        r"^[Dd]ari teks[^.]*?[:\.]\s*",
        r"^[Dd]ari data[^.]*?[:\.]\s*",
        r"^[Dd]ari referensi[^.]*?[:\.]\s*",
        
        # Frasa "Menurut..."
        r"^[Mm]enurut informasi[^.]*?[:\.]\s*",
        r"^[Mm]enurut sumber[^.]*?[:\.]\s*",
        r"^[Mm]enurut teks[^.]*?[:\.]\s*",
        
        # Frasa "Mengacu/Merujuk..."
        r"^[Mm]engacu pada[^.]*?[:\.]\s*",
        r"^[Mm]erujuk pada[^.]*?[:\.]\s*",
        r"^[Mm]elihat informasi[^.]*?[:\.]\s*",
        
        # Frasa "Sebagaimana..."
        r"^[Ss]ebagaimana disebutkan[^.]*?[:\.]\s*",
        r"^[Ss]ebagai informasi[^.]*?[:\.]\s*",
        
        # Frasa "Dalam konteks..."
        r"^[Dd]alam konteks[^.]*?[:\.]\s*",
        r"^[Dd]alam informasi[^.]*?[:\.]\s*",
        
        # Frasa "Teks/Sumber tersebut..."
        r"^[Tt]eks tersebut[^.]*?[:\.]\s*",
        r"^[Ss]umber tersebut[^.]*?[:\.]\s*",
        r"^[Ii]nformasi tersebut[^.]*?[:\.]\s*",
        
        # Frasa "Anda memberikan..."
        r"^[Yy]ang Anda berikan[^.]*?[:\.]\s*",
        r"^[Yy]ang Anda sampaikan[^.]*?[:\.]\s*",
        
        # Frasa "Berikut adalah..."
        r"^[Bb]erikut (?:adalah|merupakan)[^.]*?[:\.]\s*",
        r"^[Bb]erikut adalah lanjutan[^.]*?[:\.]\s*",
        
        # Frasa "Jawaban lanjutan..."
        r"^[Jj]awaban lanjutan[:\.]\s*",
        r"^[Mm]elanjutkan jawaban[:\.]\s*",
        r"^[Uu]ntuk melanjutkan[^.]*?[:\.]\s*",
        r"^[Pp]ada jawaban sebelumnya[^.]*?[:\.]\s*",
        
        # PERBAIKAN BARU: Frasa "Namun, sebagai..."
        r"^[Nn]amun, sebagai [^.]*?[:\.]\s*",
        r"^[Nn]amun sebagai [^.]*?[:\.]\s*",
        r"^[Nn]amun, saya [^.]*?[:\.]\s*",
        r"^[Nn]amun saya [^.]*?[:\.]\s*",
        
        # PERBAIKAN BARU: Frasa "Mohon maaf..."
        r"^[Mm]ohon maaf[^.]*?[:\.]\s*",
        r"^[Mm]aaf[^.]*?[:\.]\s*",
        
        # PERBAIKAN BARU: Frasa "Saya dapat/mampu..."
        r"^[Ss]aya (?:dapat|bisa|mampu)[^.]*?memberikan[^.]*?[:\.]\s*",
        r"^[Ss]ebagai [^.]*?saya (?:dapat|bisa|mampu)[^.]*?[:\.]\s*",
        
        # PERBAIKAN BARU: Frasa "Anda belum menyebutkan..."
        r"^[Aa]nda belum menyebutkan[^.]*?[:\.]\s*",
        r"^[Kk]arena Anda belum[^.]*?[:\.]\s*",
    ]
    
    cleaned = answer.strip()
    
    # Terapkan semua pola pembersih (lakukan beberapa kali untuk pola bersarang)
    for _ in range(3):  # Loop 3x untuk menangkap pola bersarang
        for pattern in unwanted_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    
    # Deteksi overlap dengan jawaban sebelumnya
    if previous_answer and len(previous_answer) > 30:
        prev_sentences = re.split(r'(?<=[.!?])\s+', previous_answer)
        prev_tail = " ".join(prev_sentences[-3:]).strip()
        
        if cleaned.startswith(prev_tail[:50]):
            for i in range(min(len(cleaned), len(prev_tail)), 0, -1):
                if not cleaned.startswith(prev_tail[:i]):
                    cleaned = cleaned[i:].strip()
                    break
    
    if not cleaned:
        return answer.strip()
    
    return cleaned


# ==============================
# 5. FUNGSI CORE: RETRIEVAL & GENERATION
# ==============================

def search_documents(query: str, n_results: int = 3):
    """Cari dokumen relevan dari ChromaDB."""
    try:
        query_embedding = embedder.encode(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        return list(zip(docs, metadatas, distances)) if docs else []
        
    except Exception as e:
        print(f"[Error Retrieval] {e}")
        return []


def filter_relevant_documents(docs_with_meta: list, threshold: float = None) -> list:
    """Filter dokumen berdasarkan similarity score."""
    if threshold is None:
        threshold = RELEVANCE_THRESHOLD
    
    filtered = []
    for doc, meta, distance in docs_with_meta:
        similarity = 1 / (1 + distance)
        if similarity >= threshold:
            filtered.append((doc, meta, distance, similarity))
    
    return filtered


def call_llm(prompt: str, max_tokens: int = None, timeout: int = None) -> tuple:
    """Kirim prompt ke LM Studio API."""
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
        
        is_truncated = False
        if content:
            last_char = content.rstrip()[-1] if content else ""
            if finish_reason == "length" and last_char not in ".!?\"'":
                is_truncated = True
            elif finish_reason not in ["stop", "length"] and last_char not in ".!?\"'":
                is_truncated = True
        
        return content, is_truncated
        
    except requests.exceptions.Timeout:
        return "[Error] Timeout: Coba pertanyaan yang lebih singkat.", False
    except requests.exceptions.ConnectionError:
        return "[Error] Pastikan LM Studio berjalan di http://127.0.0.1:1234", False
    except Exception as e:
        return f"[Error] {type(e).__name__}: {e}", False


def get_contextual_query(query: str) -> str:
    """Membuat query pencarian lebih kontekstual."""
    # PERBAIKAN: Gunakan resolve_references untuk mendeteksi kata referensi
    enhanced_query = resolve_references(query)
    
    # Jika query sangat pendek dan tidak ada referensi, gabungkan dengan query sebelumnya
    if enhanced_query == query:  # Tidak ada referensi yang terdeteksi
        words = query.split()
        if len(words) < 5 and conversation_history:
            last_query = conversation_history[-1].get("query", "")
            return f"{query} {last_query}"
    
    return enhanced_query


def generate_response(query: str, ground_truth: str = "") -> tuple:
    """Generate jawaban menggunakan pipeline RAG."""
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
    
    # PERBAIKAN KUNCI: Query Enhancement Pipeline
    # 1. Resolve references (hal demikian, itu, tersebut)
    # 2. Get contextual query (gabungkan dengan percakapan sebelumnya)
    # 3. Expand query (tambahkan sinonim)
    enhanced_query = get_contextual_query(query)
    expanded_query = expand_query(enhanced_query)
    
    print(f"Mencari referensi...", end="\r")
    docs_with_meta = search_documents(expanded_query, n_results=TOP_K * 2)
    print(" " * 40, end="\r")
    
    relevant_docs = filter_relevant_documents(docs_with_meta)
    
    # Bangun riwayat percakapan untuk konteks
    history_text = ""
    for turn in conversation_history:
        history_text += f"Pengguna: {turn['query']}\nAsisten: {turn['answer']}\n\n"
    
    if not relevant_docs:
        # Fallback ke pengetahuan umum
        prompt = f"""Anda adalah dokter/bidan yang ramah dan profesional.

RIWAYAT PERCAKAPAN:
{history_text}

PERTANYAAN PASIEN:
{query}

TUGAS:
Jawab pertanyaan pasien secara langsung, jelas, dan empatik menggunakan pengetahuan umum Anda tentang kesehatan ibu dan anak.

ATURAN PENTING:
1. LANGSUNG mulai dengan jawaban - JANGAN pakai pembuka seperti "Berdasarkan...", "Namun, sebagai...", "Mohon maaf...", dll.
2. JANGAN menyebut "sumber", "referensi", atau "teks".
3. Gunakan bahasa Indonesia yang natural seperti dokter yang sedang berbicara dengan pasien.
4. Pahami konteks percakapan sebelumnya untuk menjawab pertanyaan yang merujuk ke topik sebelumnya.
5. Gunakan format poin (-) untuk daftar jika diperlukan.
6. Untuk kondisi darurat, sarankan untuk segera ke Puskesmas Kuranji.

JAWABAN ANDA (langsung ke inti):"""
        
        print("Menyusun jawaban (pengetahuan umum)...", end="\r")
        answer, is_truncated = call_llm(prompt)
        print(" " * 40, end="\r")
        
        answer = clean_response(answer)
        
        if is_truncated:
            answer += "\n\n*(Jawaban terpotong. Silakan ketik 'lanjutkan' untuk melanjutkan)*"
        
        conversation_history.append({
            "query": query,
            "answer": answer,
            "context": "GENERAL_KNOWLEDGE",
            "timestamp": datetime.now().isoformat()
        })
        
        log_interaction(
            query=query,
            answer=answer,
            context="GENERAL_KNOWLEDGE",
            contexts=[],
            ground_truth=ground_truth,
            metadata={
                "mode": "general_knowledge", 
                "truncated": is_truncated,
                "enhanced_query": expanded_query
            }
        )
        
        return answer, is_truncated
    
    # Ada dokumen relevan → gunakan RAG
    docs = [d[0] for d in relevant_docs]
    context = "\n\n---\n\n".join(docs)
    
    if len(context) > 1500:
        context = truncate_context(context)
    
    # PERBAIKAN KUNCI: Prompt yang menekankan konteks percakapan
    prompt = f"""Anda adalah dokter/bidan yang ramah sedang berbicara langsung dengan pasien.

CATATAN MEDIS (untuk referensi internal Anda - JANGAN sebutkan ini ke pasien):
---
{context}
---

RIWAYAT PERCAKAPAN:
{history_text}

PERTANYAAN PASIEN SAAT INI:
{query}

TUGAS ANDA:
Jawab pertanyaan pasien secara langsung, jelas, dan empatik seperti dokter profesional.

ATURAN SANGAT PENTING:
1. PAHAMI KONTEKS PERCAKAPAN - Jika pasien menyebut "hal demikian", "itu", "tersebut", atau merujuk ke pertanyaan sebelumnya, gunakan konteks dari riwayat percakapan untuk memahami apa yang dimaksud.
2. LANGSUNG mulai menjawab - JANGAN pakai frasa seperti:
   - "Berdasarkan informasi..."
   - "Namun, sebagai asisten..."
   - "Mohon maaf..."
   - "Anda belum menyebutkan..."
   - "Dari sumber yang diberikan..."
   - "Menurut catatan..."
3. JANGAN PERNAH menyebut "sumber", "referensi", "catatan", atau "teks" dalam jawaban.
4. Gunakan bahasa Indonesia yang natural dan mudah dipahami.
5. Jika kondisi serius, sarankan untuk segera ke Puskesmas Kuranji.
6. Gunakan format poin (-) untuk daftar jika perlu.
7. Bersikaplah empatik dan profesional.

JAWABAN ANDA KE PASIEN (langsung ke inti, tanpa pembuka):"""
    
    print("Menyusun jawaban...", end="\r")
    answer, is_truncated = call_llm(prompt)
    print(" " * 40, end="\r")
    
    answer = clean_response(answer)
    
    if is_truncated:
        answer += "\n\n*(Jawaban terpotong. Silakan ketik 'lanjutkan' untuk melanjutkan)*"
    
    conversation_history.append({
        "query": query,
        "answer": answer,
        "context": context,
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
            "answer_length": len(answer),
            "avg_similarity": sum(d[3] for d in relevant_docs) / len(relevant_docs) if relevant_docs else 0,
            "enhanced_query": expanded_query
        }
    )
    
    return answer, is_truncated


def continue_response():
    """Melanjutkan jawaban RAG yang terpotong."""
    if not conversation_history:
        return "Maaf, belum ada jawaban sebelumnya yang bisa dilanjutkan. Silakan ajukan pertanyaan baru.", False
    
    last_data = conversation_history[-1]
    last_answer = last_data.get("answer", "")
    last_context = last_data.get("context", "")
    
    if not last_context or last_context == "NO_RESULTS":
        return "Jawaban sebelumnya tidak memiliki referensi yang cukup untuk dilanjutkan.", False

    prompt = f"""Anda sedang melanjutkan kalimat yang terpotong.

CATATAN MEDIS:
{last_context}

JAWABAN YANG TERPOTONG (akhiran):
"...{last_answer[-200:]}"

TUGAS: Tulis 2-3 kalimat berikutnya yang nyambung secara tata bahasa.

ATURAN WAJIB:
1. LANGSUNG mulai dengan kata pertama kalimat baru.
2. JANGAN tulis kata pembuka apapun (seperti "Berdasarkan...", "Jawaban lanjutan:", dll).
3. JANGAN ulangi kalimat yang sudah ada.

CONTOH BENAR:
Jika terpotong di: "...ibu hamil juga sering mengalami mual"
Maka tulis: "Selain itu, rasa lelah berlebih juga umum dirasakan. Pastikan ibu tetap istirahat cukup."

TULIS LANJUTANNYA SEKARANG (tanpa pembuka apapun):"""

    print("Melanjutkan jawaban...", end="\r")
    answer, is_truncated = call_llm(prompt)
    print(" " * 40, end="\r")
    
    answer = clean_response(answer, last_answer)
    
    if not answer or len(answer) < 20:
        answer = "Maaf, terjadi kesalahan saat melanjutkan jawaban. Silakan ajukan pertanyaan baru."
        is_truncated = False
    elif is_truncated:
        answer += "\n\n*(Masih terpotong. Ketik 'lanjutkan' lagi jika perlu)*"
        
    conversation_history[-1]["answer"] = last_answer + " " + answer
    
    log_interaction(
        query="[PERINTAH: LANJUTKAN]",
        answer=answer,
        context=last_context,
        contexts=[],
        metadata={"is_continuation": True, "cleaned": True}
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
    """Konversi file CSV manual menjadi dataset_evaluasi.json."""
    print("=" * 60)
    print("MODE KONVERSI CSV KE JSON")
    print("=" * 60)
    
    if not CSV_INPUT_FILE.exists():
        print(f"[ERROR] File {CSV_INPUT_FILE} tidak ditemukan.")
        return
    
    dataset = []
    try:
        with open(CSV_INPUT_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "question" not in row or "ground_truth" not in row:
                    print("[WARNING] File CSV harus memiliki kolom 'question' dan 'ground_truth'")
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
        print("=" * 60)
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan file JSON: {e}")


# ==============================
# 8. MODE EVALUASI
# ==============================

def run_evaluation():
    """Menjalankan chatbot secara otomatis menggunakan dataset evaluasi."""
    print("\n" + "="*60)
    print("MEMULAI MODE EVALUASI RAG")
    print("="*60)

    if not DATASET_EVAL_FILE.exists():
        print(f"[ERROR] File dataset evaluasi tidak ditemukan: {DATASET_EVAL_FILE}")
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

        expanded_query = expand_query(query)
        docs_with_meta = search_documents(expanded_query, n_results=TOP_K * 2)
        relevant_docs = filter_relevant_documents(docs_with_meta)
        
        if not relevant_docs:
            prompt = f"""Anda adalah dokter/bidan yang ramah.
Pengguna bertanya: "{query}"
Jawablah berdasarkan pengetahuan umum Anda.
Gunakan bahasa Indonesia yang jelas.
LANGSUNG jawab tanpa pembuka.

JAWABAN:"""
            answer, _ = call_llm(prompt)
            answer = clean_response(answer)
            docs = []
            context = "GENERAL_KNOWLEDGE"
        else:
            docs = [d[0] for d in relevant_docs]
            context = "\n\n---\n\n".join(docs)
            if len(context) > 1500:
                context = truncate_context(context)
                
            prompt = f"""Anda adalah dokter/bidan yang ramah.

CATATAN MEDIS (referensi internal - JANGAN sebutkan):
{context}

PERTANYAAN PASIEN:
{query}

Jawab langsung tanpa pembuka seperti "Berdasarkan...", "Dari sumber...", dll.
Gunakan bahasa Indonesia yang jelas.

JAWABAN:"""
            answer, _ = call_llm(prompt)
            answer = clean_response(answer)

        log_interaction(
            query=query,
            answer=answer,
            context=context,
            contexts=docs,
            ground_truth=ground_truth,
            metadata={
                "doc_count": len(docs),
                "evaluation_mode": True
            }
        )
        print(f"   [OK] Jawaban & ground_truth tersimpan.\n")

    print("="*60)
    print(f"[SELESAI] {len(dataset)} pertanyaan telah diuji.")
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
    print("  /exit     -> Keluar dari chatbot")
    print("  /logs     -> Lihat riwayat interaksi")
    print("  /help     -> Tampilkan panduan ini")
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
            
            if query.lower() in ["lanjutkan", "lanjut", "continue"]:
                print("Chatbot: ", end="", flush=True)
                answer, was_truncated = continue_response()
                print(answer)
                
                if was_truncated:
                    print("\nTips: Jawaban masih terpotong. Ketik 'lanjutkan' lagi jika perlu.")
                print("\n" + "-" * 60)
                continue
            
            print("Chatbot: ", end="", flush=True)
            answer, was_truncated = generate_response(query)
            print(answer)
            
            if was_truncated:
                print("\nTips: Jawaban terpotong. Ketik 'lanjutkan' untuk melanjutkan.")
            
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
    else:
        main()