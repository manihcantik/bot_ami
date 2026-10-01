# ==============================
# 1. IMPORTS
# ==============================

import os
import sys
import subprocess
import json
import csv
import requests
import re
import warnings
import time
import uuid
from datetime import datetime
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
import chromadb

from config.config import (
    PROJECT_ROOT,
    SCENARIO_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    EMBEDDING_MODEL,
    DB_PATH,
    COLLECTION_NAME,
    LM_API_URL,
    LLM_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    TIMEOUT_SECONDS,
    MAX_HISTORY,
    LOG_FILE,
    GUARDRAIL_LOG_FILE,
    DATASET_EVAL_FILE,
    CSV_INPUT_FILE,
    RELEVANCE_THRESHOLD,
    GUARDRAIL_MAX_CONTEXT_CHARS,
)

from config.scenarios import (
    SCENARIOS,
    get_scenario,
    get_all_scenarios,
)

from app.guardrails import (
    NO_CONTEXT_MESSAGE,
    append_disclaimer,
    format_sources,
    inspect_query,
)

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")


# ==============================
# 2. INISIALISASI GLOBAL
# ==============================

print(
    f"[1/3] Loading embedding model "
    f"({EMBEDDING_MODEL})..."
)

embedder = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    f"[2/3] Menghubungkan ke ChromaDB "
    f"di {DB_PATH}..."
)

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

print(
    "[3/3] Menyiapkan memori percakapan..."
)

conversation_history = deque(
    maxlen=MAX_HISTORY
)

last_retrieval_error = ""

print(
    f"\nChatbot siap digunakan! "
    f"(Skenario: {SCENARIO_NAME})\n"
)


# ==============================
# 3. FUNGSI HELPER
# ==============================

def log_interaction(
    query: str,
    answer: str,
    context: str = "",
    metadata: dict = None,
    contexts: list = None,
    ground_truth: str = "",
):
    """
    Menyimpan interaksi ke file log untuk evaluasi.
    """

    metadata = metadata or {}

    retrieved_contexts = (
        contexts
        if contexts is not None
        else []
    )

    entry = {
        "interaction_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),

        "scenario": SCENARIO_NAME,

        "rag_config": {
            "chunk_size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
        },

        "query": query,
        "answer": answer,

        "contexts": retrieved_contexts,
        "context": context,
        "ground_truth": ground_truth,

        "user_input": query,
        "retrieved_contexts": retrieved_contexts,
        "reference": ground_truth,

        "context_preview": (
            context[:200] + "..."
            if len(context) > 200
            else context
        ),

        "metadata": metadata,
    }

    try:

        Path(LOG_FILE).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            LOG_FILE,
            "a",
            encoding="utf-8-sig"
        ) as f:

            f.write(
                json.dumps(
                    entry,
                    ensure_ascii=False
                )
                + "\n"
            )

    except Exception as e:

        print(
            f"[Warning] Gagal menyimpan log: {e}"
        )


def log_guardrail_event(
    query: str,
    decision,
    answer: str
) -> None:

    """
    Menyimpan aktivasi guardrail
    terpisah dari log evaluasi normal.
    """

    entry = {
        "timestamp": datetime.now().isoformat(),
        "scenario": SCENARIO_NAME,
        "query_length": len(query),
        "reason": decision.reason,
        "action": decision.action,
        "answer": answer,
    }

    try:

        Path(
            GUARDRAIL_LOG_FILE
        ).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            GUARDRAIL_LOG_FILE,
            "a",
            encoding="utf-8-sig"
        ) as f:

            f.write(
                json.dumps(
                    entry,
                    ensure_ascii=False
                )
                + "\n"
            )

    except Exception as e:

        print(
            f"[Warning] Gagal menyimpan "
            f"log guardrail: {e}"
        )


def detect_intent(query: str) -> dict:
    """
    Deteksi intent dasar:
    greeting dan exit.
    """

    q = query.lower().strip()

    clean_q = re.sub(
        r"[^\w\s]",
        "",
        q
    ).strip()

    greeting_words = [
        "halo",
        "hai",
        "hello",
        "hi",
        "assalamualaikum",
        "selamat pagi",
        "selamat siang",
        "selamat sore",
        "selamat malam",
    ]

    is_only_greeting = (
        clean_q in greeting_words
    )

    exit_words = [
        "exit",
        "quit",
        "keluar",
        "bye",
        "selesai",
        "terima kasih",
    ]

    is_exit = (
        clean_q in exit_words
    )

    return {
        "is_greeting": is_only_greeting,
        "is_exit": is_exit,
    }


def truncate_context(
    context: str,
    max_chars: int = 2000
) -> str:

    """
    Fungsi utilitas pemotongan context.

    Tidak digunakan pada context utama eksperimen.
    """

    if len(context) <= max_chars:
        return context

    cut_text = context[:max_chars]

    last_period = cut_text.rfind(".")

    if last_period > max_chars * 0.8:

        return (
            cut_text[:last_period + 1]
            + "\n\n[...konten dipotong...]"
        )

    return (
        cut_text
        + "\n\n[...konten dipotong...]"
    )


def expand_query(query: str) -> str:
    """
    Menambahkan istilah domain untuk memperbaiki
    recall tanpa menambahkan fakta ke jawaban.
    """

    normalized = query.lower()
    expansion_terms = []

    if (
        re.search(r"\b(?:perkembangan|pertumbuhan)\b", normalized)
        and re.search(r"\b(?:anak|bayi|balita)\b", normalized)
    ):
        expansion_terms.extend(
            ["milestone", "motorik", "bahasa", "sosial"]
        )

    if re.search(
        r"\b(?:pelayanan|layanan|dibawa)\b",
        normalized
    ) and re.search(
        r"\b(?:anak|bayi|balita|posyandu|puskesmas)\b",
        normalized
    ):
        expansion_terms.extend(
            [
                "pemantauan pertumbuhan",
                "pemantauan perkembangan",
                "kelas ibu balita",
                "vitamin A",
                "imunisasi",
            ]
        )

    if re.search(r"\b(?:ttd|tambah darah)\b", normalized):
        expansion_terms.extend(
            ["tablet tambah darah", "zat besi", "asam folat", "anemia"]
        )

    if re.search(r"\b(?:persalinan|melahirkan)\b", normalized):
        expansion_terms.extend(
            ["kontraksi", "his", "lendir darah", "ketuban"]
        )

    if "mpasi" in normalized or "mp asi" in normalized:
        expansion_terms.extend(
            ["makanan pendamping ASI", "protein hewani", "tekstur"]
        )

    if not expansion_terms:
        return query

    return f"{query} {' '.join(expansion_terms)}"


def normalize_query(query: str) -> str:
    """
    Normalisasi typo umum dan singkatan KIA
    yang tidak mengubah maksud.
    """

    replacements = {
        "kehamillan": "kehamilan",
        "imunisasii": "imunisasi",
        "menyusuii": "menyusui",
        "pemeriksaaan": "pemeriksaan",
        "persalinaan": "persalinan",
        "bayii": "bayi",
        "balitaa": "balita",
        "mp asi": "mpasi",
    }

    normalized = query

    for source, target in replacements.items():

        normalized = re.sub(
            rf"\b{re.escape(source)}\b",
            target,
            normalized,
            flags=re.IGNORECASE
        )

    return normalized


def resolve_references(query: str) -> str:
    """
    Deteksi kata referensi dan gabungkan
    dengan konteks sebelumnya.
    """

    q = query.lower().strip()

    reference_words = [
        "hal demikian",
        "hal itu",
        "hal tersebut",
        "itu",
        "tersebut",
        "tadi",
        "sebelumnya",
        "yang itu",
        "yang tadi",
        "hal yang sama",
        "gejala tersebut",
        "gejala itu",
        "kondisi tersebut",
        "kondisi itu",
        "penyakit itu",
        "penyakit tersebut",
        "masalah itu",
        "masalah tersebut",
    ]

    has_reference = any(
        ref in q
        for ref in reference_words
    )

    if has_reference and conversation_history:

        last_turn = conversation_history[-1]

        last_query = last_turn.get(
            "query",
            ""
        )

        enhanced_query = (
            f"{query} {last_query}"
        )

        if (
            "halangan" in q
            or "mens" in q
            or "haid" in q
        ):

            enhanced_query += (
                " kehamilan menstruasi haid"
            )

        return enhanced_query

    return query


def clean_response(
    answer: str,
    previous_answer: str = ""
) -> str:

    """
    Membersihkan frasa pembuka
    yang tidak diinginkan.
    """

    if not answer:
        return answer

    unwanted_patterns = [
        r"^[Bb]erdasarkan "
        r"(?:informasi|teks|sumber|data|referensi|catatan)"
        r"[^.]*[:.]\s*",

        r"^[Dd]ari "
        r"(?:informasi|sumber|teks|data)"
        r"[^.]*[:.]\s*",

        r"^[Mm]enurut "
        r"(?:informasi|sumber|teks)"
        r"[^.]*[:.]\s*",

        r"^[Uu]ntuk menjawab pertanyaan ini"
        r"[^.]*[:.]\s*",

        r"^[Mm]enjawab pertanyaan"
        r"[^.]*[:.]\s*",

        r"^[Tt]erkait dengan pertanyaan"
        r"[^.]*[:.]\s*",

        r"^[Jj]awaban untuk pertanyaan"
        r"[^.]*[:.]\s*",
    ]

    cleaned = answer.strip()

    for pattern in unwanted_patterns:

        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE
        ).strip()

    if previous_answer and len(previous_answer) > 30:

        prev_sentences = re.split(
            r"(?<=[.!?])\s+",
            previous_answer
        )

        prev_tail = " ".join(
            prev_sentences[-3:]
        ).strip()

        if cleaned.startswith(
            prev_tail[:50]
        ):

            for i in range(
                min(
                    len(cleaned),
                    len(prev_tail)
                ),
                0,
                -1
            ):

                if not cleaned.startswith(
                    prev_tail[:i]
                ):

                    cleaned = (
                        cleaned[i:]
                        .strip()
                    )

                    break

    if not cleaned:
        return answer.strip()

    return cleaned


# ==============================
# 4. CORE RETRIEVAL & GENERATION
# ==============================

def search_documents(
    query: str,
    n_results: int = None
):
    """
    Mencari dokumen relevan dari ChromaDB.

    TOP_K langsung digunakan sebagai jumlah
    hasil retrieval sesuai rancangan eksperimen.
    """

    global last_retrieval_error

    last_retrieval_error = ""

    if n_results is None:

        # Sesuai rancangan eksperimen:
        # top-k = 3, 5, atau 7.
        n_results = TOP_K

    try:

        query_embedding = (
            embedder
            .encode(
                query,
                normalize_embeddings=True
            )
            .tolist()
        )

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        docs = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        if not docs:
            return []

        return list(
            zip(
                docs,
                metadatas,
                distances
            )
        )

    except Exception as e:

        last_retrieval_error = str(e)

        print(
            f"[Error Retrieval] {e}"
        )

        return []


def filter_relevant_documents(
    docs_with_meta: list,
    top_k: int = None
) -> list:

    """
    Mengambil hasil retrieval sesuai top-k.

    Tidak menggunakan threshold similarity
    untuk membuang dokumen.

    Jumlah context tetap konsisten dengan
    rancangan S1-S9.
    """

    if top_k is None:
        top_k = TOP_K

    if not docs_with_meta:
        return []

    selected = docs_with_meta[
        :top_k
    ]

    result = []

    for doc, meta, distance in selected:

        similarity = (
            1 / (1 + distance)
            if distance is not None
            else 0
        )

        result.append(
            (
                doc,
                meta,
                distance,
                similarity,
            )
        )

    return result


def call_llm(
    prompt: str,
    max_tokens: int = None,
    timeout: int = None
) -> tuple:

    """
    Mengirim prompt ke LM Studio API.
    """

    max_tokens = (
        max_tokens
        or MAX_TOKENS
    )

    timeout = (
        timeout
        or TIMEOUT_SECONDS
    )

    try:

        response = requests.post(
            LM_API_URL,

            json={
                "model": LLM_MODEL,

                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],

                "temperature": TEMPERATURE,
                "max_tokens": max_tokens,
            },

            timeout=timeout,

            headers={
                "Content-Type":
                    "application/json"
            },
        )

        response.raise_for_status()

        result = response.json()

        content = (
            result["choices"][0]
            ["message"]["content"]
            .strip()
        )

        finish_reason = (
            result["choices"][0]
            .get(
                "finish_reason",
                ""
            )
        )

        is_truncated = False

        if content:

            last_char = (
                content.rstrip()[-1]
                if content
                else ""
            )

            if (
                finish_reason == "length"
                and last_char
                not in ".!?'\""
            ):

                is_truncated = True

            elif (
                finish_reason
                not in ["stop", "length"]
                and last_char
                not in ".!?'\""
            ):

                is_truncated = True

        return content, is_truncated

    except requests.exceptions.Timeout:

        return (
            "[Error] Timeout: "
            "Coba pertanyaan yang lebih singkat.",
            False,
        )

    except requests.exceptions.ConnectionError:

        return (
            "[Error] Pastikan LM Studio "
            f"berjalan di {LM_API_URL}",
            False,
        )

    except Exception as e:

        return (
            f"[Error] "
            f"{type(e).__name__}: {e}",
            False,
        )


def get_contextual_query(
    query: str
) -> str:

    """
    Membuat query pencarian lebih kontekstual.

    Riwayat percakapan hanya digunakan apabila
    pertanyaan memang mengandung kata rujukan
    terhadap percakapan sebelumnya.
    """

    contextual_query = resolve_references(
        query
    )

    return expand_query(
        contextual_query
    )


def generate_response(
    query: str,
    ground_truth: str = ""
) -> tuple:

    """
    Generate jawaban menggunakan pipeline RAG.

    Alur:
    1. Deteksi intent/greeting
    2. Pemeriksaan guardrail
    3. Normalisasi query
    4. Contextual query
    5. Retrieval ChromaDB
    6. Filter top-k
    7. Penyusunan context
    8. Generate jawaban menggunakan LLM
    9. Disclaimer dan sumber
    10. Memory
    11. Logging evaluasi
    """

    started_at = time.perf_counter()

    def log_metadata(
        metadata: dict | None = None
    ) -> dict:

        result = dict(
            metadata or {}
        )

        result["processing_ms"] = round(
            (
                time.perf_counter()
                - started_at
            ) * 1000,
            2
        )

        return result

    # ============================================================
    # 1. DETEKSI INTENT
    # ============================================================

    intents = detect_intent(query)

    # Greeting harus diperiksa SEBELUM guardrail.
    # Jika tidak, greeting seperti "halo" dapat dianggap
    # sebagai pertanyaan di luar topik KIA.

    if intents["is_greeting"]:

        answer = append_disclaimer(
            "Halo! Saya adalah asisten "
            "edukasi kesehatan ibu dan anak.\n\n"
            "Saya siap membantu memberikan "
            "informasi seputar:\n\n"
            "- Kehamilan, persiapan persalinan, "
            "dan masa nifas\n"
            "- Tumbuh kembang bayi dan anak\n"
            "- Imunisasi dan kesehatan anak\n"
            "- Gizi ibu hamil, menyusui, "
            "serta nutrisi anak\n"
            "- Keluhan umum pada ibu dan anak\n\n"
            "Silakan ajukan pertanyaan "
            "yang ingin Anda ketahui."
        )

        conversation_history.append({
            "query": query,
            "answer": answer,
            "context": "GREETING",
            "timestamp": datetime.now().isoformat(),
        })

        log_interaction(
            query=query,
            answer=answer,
            context="GREETING",
            contexts=[],
            ground_truth=ground_truth,
            metadata=log_metadata({
                "intent": "greeting",
                "mode": "greeting",
                "guardrail_action": "allow",
                "retrieval_skipped": True,
                "retrieval_count": 0,
            }),
        )

        return answer, False

    # ============================================================
    # 2. GUARDRAIL
    # ============================================================

    decision = inspect_query(query)

    if decision.action != "allow":

        answer = append_disclaimer(
            decision.response,
            sensitive=(
                decision.action
                == "emergency"
            ),
        )

        # Simpan event guardrail
        log_guardrail_event(
            query,
            decision,
            answer
        )

        # Guardrail TIDAK masuk ke retrieval.
        log_interaction(
            query=query,
            answer=answer,
            context="GUARDRAIL",
            contexts=[],
            ground_truth=ground_truth,
            metadata=log_metadata({
                "guardrail_active": True,
                "reason": decision.reason,
                "action": decision.action,
                "retrieval_skipped": True,
                "retrieval_count": 0,
                "doc_count": 0,
            }),
        )

        return answer, False

    # ============================================================
    # 3. NORMALISASI QUERY
    # ============================================================

    normalized_query = normalize_query(
        query
    )

    # ============================================================
    # 4. KONTEKS PERCAKAPAN
    # ============================================================

    retrieval_query = get_contextual_query(
        normalized_query
    )

    # ============================================================
    # 5. RETRIEVAL
    # ============================================================

    print(
        "Mencari referensi...",
        end="\r"
    )

    docs_with_meta = search_documents(
        retrieval_query
    )

    print(
        " " * 40,
        end="\r"
    )

    # Jumlah dokumen mengikuti TOP_K
    # sesuai konfigurasi skenario.
    relevant_docs = (
        filter_relevant_documents(
            docs_with_meta,
            TOP_K
        )
    )

    # ============================================================
    # 6. HISTORY
    # ============================================================

    history_text = ""

    reference_words = [
        "hal demikian",
        "hal itu",
        "hal tersebut",
        "itu",
        "tersebut",
        "tadi",
        "sebelumnya",
        "yang itu",
        "yang tadi",
        "hal yang sama",
        "gejala tersebut",
        "gejala itu",
        "kondisi tersebut",
        "kondisi itu",
        "penyakit itu",
        "penyakit tersebut",
        "masalah itu",
        "masalah tersebut",
    ]

    query_lower = query.lower()

    use_history = any(
        ref in query_lower
        for ref in reference_words
    )

    if use_history:

        for turn in conversation_history:

            history_text += (
                f"Pengguna: "
                f"{turn['query']}\n"
                f"Asisten: "
                f"{turn['answer']}\n\n"
            )

    if not history_text:

        history_text = (
            "Tidak ada riwayat percakapan "
            "yang diperlukan."
        )

    # ============================================================
    # 7. CEK APAKAH CONTEXT DITEMUKAN
    # ============================================================

    if not relevant_docs:

        if last_retrieval_error:

            answer = (
                "Maaf, basis pengetahuan "
                "sedang tidak dapat diakses. "
                "Silakan coba lagi nanti atau "
                "hubungi tenaga kesehatan."
            )

            mode = "retrieval_error"

        else:

            answer = (
                NO_CONTEXT_MESSAGE
            )

            mode = "no_relevant_context"

        answer = append_disclaimer(
            answer
        )

        log_interaction(
            query=query,
            answer=answer,
            context="NO_RESULTS",
            contexts=[],
            ground_truth=ground_truth,
            metadata=log_metadata({
                "fallback": True,
                "mode": mode,
                "original_query": query,
                "normalized_query": normalized_query,
                "retrieval_query": retrieval_query,
                "top_k": TOP_K,
                "chunk_size": CHUNK_SIZE,
                "overlap": CHUNK_OVERLAP,
                "retrieval_skipped": False,
                "retrieval_count": 0,
                "doc_count": 0,
            }),
        )

        return answer, False

    # ============================================================
    # 8. FORMAT CONTEXT
    # ============================================================

    docs = [
        d[0]
        for d in relevant_docs
    ]

    # Context diberikan ke LLM secara utuh.
    # Tidak dipotong menjadi 2000 karakter.

    context = (
        "\n\n---\n\n".join(docs)
    )

    citations = format_sources(
        relevant_docs
    )

    # ============================================================
    # 9. PROMPT
    # ============================================================

    prompt = f"""Anda adalah asisten edukasi Kesehatan Ibu dan Anak (KIA) yang ramah.

DOKUMEN SUMBER:

---

{context}

---

RIWAYAT PERCAKAPAN:

{history_text}

PERTANYAAN PENGGUNA:

{query}

TUGAS:

Jawab pertanyaan pengguna secara langsung, jelas,
terstruktur, dan mudah dipahami.

ATURAN:

1. Gunakan hanya informasi yang terdapat dalam DOKUMEN SUMBER.

2. Jangan menggunakan pengetahuan dari luar DOKUMEN SUMBER.

3. Jangan membuat informasi yang tidak terdapat dalam
   DOKUMEN SUMBER.

4. Jika DOKUMEN SUMBER tidak memiliki informasi yang
   cukup untuk menjawab pertanyaan, katakan:

   "Maaf, informasi tersebut belum tersedia dalam
   sumber yang saya gunakan."

5. Jangan mendiagnosis penyakit.

6. Jangan memberikan obat, dosis obat, atau instruksi
   pengobatan yang tidak terdapat dalam dokumen.

7. Jangan memberikan kepastian medis mutlak.

8. Jawab langsung sesuai pertanyaan pengguna.

9. Jika pertanyaan meminta "apa saja", "sebutkan", atau
   pertanyaan cakupan umum, kumpulkan seluruh poin yang
   relevan dan didukung oleh DOKUMEN SUMBER.

10. Jangan memasukkan poin dari topik lain hanya karena
    poin tersebut kebetulan ada di DOKUMEN SUMBER.

11. Untuk pertanyaan tentang angka, usia, waktu, frekuensi,
    jadwal, atau tahapan, prioritaskan informasi spesifik
    tersebut jika tersedia di DOKUMEN SUMBER.

12. Jika DOKUMEN SUMBER hanya mendukung sebagian jawaban,
    jawab bagian yang didukung dan jangan mengarang sisanya.

13. Jangan mengulang pertanyaan pengguna.

14. Jangan membahas proses internal sistem.

15. Jangan memberikan diagnosis, obat, dosis, atau instruksi
    pengobatan tertentu.

JAWABAN:"""

    # ============================================================
    # 10. GENERATE RESPONSE
    # ============================================================

    print(
        "Menyusun jawaban...",
        end="\r"
    )

    answer, is_truncated = call_llm(
        prompt
    )

    print(
        " " * 40,
        end="\r"
    )

    answer = clean_response(
        answer
    )

    # ============================================================
    # 11. ERROR RESPONSE
    # ============================================================

    if (
        answer.startswith("[Error]")
        or not answer.strip()
    ):

        answer = (
            "Maaf, model bahasa sedang "
            "tidak dapat memberikan jawaban. "
            "Silakan coba lagi nanti atau "
            "hubungi tenaga kesehatan."
        )

        is_truncated = False

        answer = append_disclaimer(
            answer
        )

    else:

        answer = append_disclaimer(
            answer
        )

        if citations:

            answer += (
                f"\n\nSumber: {citations}"
            )

    # ============================================================
    # 12. TRUNCATED RESPONSE
    # ============================================================

    if is_truncated:

        answer += (
            "\n\n*(Jawaban terpotong. "
            "Silakan ketik 'lanjutkan' "
            "untuk melanjutkan)*"
        )

    # ============================================================
    # 13. MEMORY
    # ============================================================

    conversation_history.append({
        "query": query,
        "answer": answer,
        "context": context,
        "timestamp": datetime.now().isoformat(),
    })

    # ============================================================
    # 14. LOGGING EVALUASI
    # ============================================================

    log_interaction(
        query=query,
        answer=answer,

        # Context sama dengan yang digunakan LLM.
        context=context,

        contexts=docs,

        ground_truth=ground_truth,

        metadata=log_metadata({

            "intent": "rag",

            "guardrail_action": "allow",

            "retrieval_skipped": False,

            "doc_count": len(docs),

            "retrieval_count": len(docs),

            "top_k": TOP_K,

            "chunk_size": CHUNK_SIZE,

            "overlap": CHUNK_OVERLAP,

            "truncated": is_truncated,

            "answer_length": len(answer),

            "avg_similarity": (
                sum(
                    d[3]
                    for d in relevant_docs
                )
                / len(relevant_docs)
                if relevant_docs
                else 0
            ),

            "retrieval_distances": [
                d[2]
                for d in relevant_docs
            ],

            "original_query": query,

            "normalized_query": normalized_query,

            "retrieval_query": retrieval_query,

        }),
    )

    return answer, is_truncated


# ==============================
# 5. CONTINUE RESPONSE
# ==============================

def continue_response():

    """
    Melanjutkan jawaban RAG
    yang terpotong.
    """

    if not conversation_history:

        return (
            "Maaf, belum ada jawaban "
            "sebelumnya yang bisa dilanjutkan. "
            "Silakan ajukan pertanyaan baru.",
            False,
        )

    last_data = (
        conversation_history[-1]
    )

    last_answer = (
        last_data.get(
            "answer",
            ""
        )
    )

    last_context = (
        last_data.get(
            "context",
            ""
        )
    )

    if (
        not last_context
        or last_context == "NO_RESULTS"
    ):

        return (
            "Jawaban sebelumnya tidak "
            "memiliki referensi yang cukup "
            "untuk dilanjutkan.",
            False,
        )

    prompt = f"""Anda sedang melanjutkan jawaban yang terpotong.

CATATAN MEDIS:

{last_context}

JAWABAN YANG TERPOTONG:

"...{last_answer[-200:]}"

TUGAS:

Tulis kelanjutan jawaban yang nyambung
secara tata bahasa dan tetap berdasarkan
catatan medis.

ATURAN:

1. Jangan mengulang kalimat yang sudah ada.
2. Jangan menambahkan informasi di luar
   catatan medis.
3. Gunakan bahasa Indonesia yang mudah dipahami.

TULIS LANJUTANNYA:"""

    print(
        "Melanjutkan jawaban...",
        end="\r"
    )

    answer, is_truncated = call_llm(
        prompt
    )

    print(
        " " * 40,
        end="\r"
    )

    answer = clean_response(
        answer,
        last_answer
    )

    if (
        not answer
        or len(answer) < 20
    ):

        answer = (
            "Maaf, terjadi kesalahan "
            "saat melanjutkan jawaban. "
            "Silakan ajukan pertanyaan baru."
        )

        is_truncated = False

    elif is_truncated:

        answer += (
            "\n\n*(Masih terpotong. "
            "Ketik 'lanjutkan' lagi jika perlu)*"
        )

    answer = append_disclaimer(
        answer
    )

    conversation_history[-1][
        "answer"
    ] = (
        last_answer
        + " "
        + answer
    )

    log_interaction(
        query="[PERINTAH: LANJUTKAN]",
        answer=answer,
        context=last_context,

        contexts=(
            [last_context]
            if last_context
            and last_context != "NO_RESULTS"
            else []
        ),

        metadata={
            "is_continuation": True,
            "cleaned": True,
        },
    )

    return answer, is_truncated


# ==============================
# 6. UTILITAS
# ==============================

def add_document(
    doc_id: str,
    text: str,
    metadata: dict = None
) -> bool:

    """
    Menambahkan dokumen baru
    ke ChromaDB.
    """

    try:

        embedding = (
            embedder
            .encode(
                text,
                normalize_embeddings=True
            )
            .tolist()
        )

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[
                metadata or {}
            ],
        )

        print(
            f"OK: Dokumen '{doc_id}' "
            "berhasil ditambahkan."
        )

        return True

    except Exception as e:

        print(
            f"Gagal menambah dokumen: {e}"
        )

        return False


def show_logs(n: int = 5):

    """
    Menampilkan n entri log terakhir.
    """

    try:

        with open(
            LOG_FILE,
            "r",
            encoding="utf-8-sig"
        ) as f:

            lines = f.readlines()

        if not lines:

            print(
                "Belum ada log yang tersimpan."
            )

            return

        print(
            f"\n{n} interaksi terakhir:\n"
            + "-" * 60
        )

        for i, line in enumerate(
            reversed(lines[-n:]),
            1
        ):

            entry = json.loads(
                line.strip()
            )

            print(
                f"{i}. "
                f"[{entry['timestamp'][-8:]}] "
                f"{entry['query'][:50]}..."
            )

            print(
                f"   -> "
                f"{entry['answer'][:80]}"
                f"{'...' if len(entry['answer']) > 80 else ''}\n"
            )

    except FileNotFoundError:

        print(
            "File log belum ditemukan."
        )

    except Exception as e:

        print(
            f"Error membaca log: {e}"
        )


# ==============================
# 7. CSV → JSON
# ==============================

def convert_csv_to_json():

    """
    Konversi questions.csv
    menjadi dataset_evaluasi.json.
    """

    print("=" * 60)
    print("MODE KONVERSI CSV KE JSON")
    print("=" * 60)

    input_file = (
        Path(CSV_INPUT_FILE)
        .resolve()
    )

    output_file = (
        Path(DATASET_EVAL_FILE)
        .resolve()
    )

    print(
        f"[INFO] File CSV : {input_file}"
    )

    print(
        f"[INFO] File JSON : {output_file}"
    )

    if not input_file.exists():

        print(
            "\n[ERROR] File CSV tidak ditemukan!"
        )

        print(
            f"Lokasi:\n{input_file}"
        )

        return

    try:

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    except Exception as e:

        print(
            f"\n[ERROR] Gagal membuat "
            f"folder output: {e}"
        )

        return

    dataset = []

    try:

        with open(
            input_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            if reader.fieldnames is None:

                print(
                    "[ERROR] CSV tidak memiliki header."
                )

                return

            required_columns = {
                "question",
                "ground_truth"
            }

            if not required_columns.issubset(
                set(reader.fieldnames)
            ):

                print(
                    "[ERROR] CSV harus memiliki "
                    "kolom 'question' dan "
                    "'ground_truth'."
                )

                print(
                    f"Kolom ditemukan: "
                    f"{reader.fieldnames}"
                )

                return

            for row_number, row in enumerate(
                reader,
                start=2
            ):

                question = (
                    row.get("question")
                    or ""
                ).strip()

                ground_truth = (
                    row.get("ground_truth")
                    or ""
                ).strip()

                if not question or not ground_truth:

                    print(
                        f"[WARNING] Baris "
                        f"{row_number} dilewati "
                        "karena question/"
                        "ground_truth kosong."
                    )

                    continue

                dataset.append({
                    "question": question,
                    "ground_truth": ground_truth,
                })

    except Exception as e:

        print(
            "[ERROR] Gagal membaca CSV:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        return

    if not dataset:

        print(
            "\n[ERROR] Tidak ada data valid "
            "di CSV."
        )

        return

    print(
        f"\n[OK] Data valid: "
        f"{len(dataset)} pertanyaan"
    )

    conversation_history.clear()

    try:

        with open(
            output_file,
            "w",
            encoding="utf-8-sig"
        ) as f:

            json.dump(
                dataset,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            "\n" + "=" * 60
        )

        print(
            "[OK] KONVERSI BERHASIL"
        )

        print(
            f"[OK] Jumlah pertanyaan : "
            f"{len(dataset)}"
        )

        print(
            f"[OK] File JSON : "
            f"{output_file}"
        )

        print(
            "=" * 60
        )

    except Exception as e:

        print(
            "\n[ERROR] Gagal menyimpan "
            "file JSON:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )


# ==============================
# 8. PENGATURAN SKENARIO RAG
# ==============================

def configure_scenario_runtime(
    scenario_id
):

    """
    Mengaktifkan konfigurasi satu skenario
    dalam proses yang sedang berjalan.

    Tidak membuat subprocess dan tidak
    me-load embedding model lagi.
    """

    global SCENARIO_NAME
    global CHUNK_SIZE
    global CHUNK_OVERLAP
    global TOP_K
    global DB_PATH
    global LOG_FILE
    global GUARDRAIL_LOG_FILE
    global client
    global collection
    global conversation_history

    scenario_id = (
        str(scenario_id)
        .upper()
        .strip()
    )

    scenario = get_scenario(
        scenario_id
    )

    if not scenario:

        print(
            f"[ERROR] Skenario "
            f"'{scenario_id}' tidak ditemukan."
        )

        print(
            "Skenario tersedia: "
            + ", ".join(
                SCENARIOS.keys()
            )
        )

        return None

    scenario_db = (
        PROJECT_ROOT
        / "vectorstore"
        / scenario_id
    )

    if not scenario_db.exists():

        print(
            f"[ERROR] Vectorstore "
            f"skenario {scenario_id} "
            "tidak ditemukan:"
        )

        print(
            f"       {scenario_db}"
        )

        print(
            "Buat/build vectorstore "
            "skenario tersebut terlebih dahulu."
        )

        return None

    scenario_result_dir = (
        PROJECT_ROOT
        / "results"
        / scenario_id
    )

    scenario_result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    SCENARIO_NAME = scenario_id

    CHUNK_SIZE = scenario[
        "chunk_size"
    ]

    CHUNK_OVERLAP = scenario[
        "overlap"
    ]

    TOP_K = scenario[
        "top_k"
    ]

    # Validasi sesuai rancangan BAB III.

    if CHUNK_SIZE not in (
        300,
        500,
        700
    ):

        raise ValueError(
            "chunk_size tidak sesuai "
            f"rancangan eksperimen: "
            f"{CHUNK_SIZE}"
        )

    if CHUNK_OVERLAP != 50:

        raise ValueError(
            "overlap tidak sesuai "
            "rancangan eksperimen: "
            f"{CHUNK_OVERLAP}"
        )

    if TOP_K not in (
        3,
        5,
        7
    ):

        raise ValueError(
            "top_k tidak sesuai "
            "rancangan eksperimen: "
            f"{TOP_K}"
        )

    DB_PATH = str(
        scenario_db
    )

    LOG_FILE = str(
        scenario_result_dir
        / "chatbot_logs.jsonl"
    )

    GUARDRAIL_LOG_FILE = str(
        scenario_result_dir
        / "guardrail_logs.jsonl"
    )

    print(
        f"[INFO] Menghubungkan "
        f"ChromaDB skenario "
        f"{scenario_id}..."
    )

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = (
        client
        .get_or_create_collection(
            name=COLLECTION_NAME
        )
    )

    conversation_history.clear()

    return scenario


def run_single_scenario(
    scenario_id
):

    scenario = configure_scenario_runtime(
        scenario_id
    )

    if not scenario:
        return False

    scenario_id = (
        str(scenario_id)
        .upper()
        .strip()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"SKENARIO {scenario_id}"
    )

    print(
        "=" * 70
    )

    print(
        f"Chunk Size : "
        f"{scenario['chunk_size']}"
    )

    print(
        f"Overlap    : "
        f"{scenario['overlap']}"
    )

    print(
        f"Top-K      : "
        f"{scenario['top_k']}"
    )

    print(
        f"Keterangan : "
        f"{scenario['desc']}"
    )

    print(
        "=" * 70
    )

    return run_dataset()


def run_dataset_scenario(
    scenario_id
):

    scenario = configure_scenario_runtime(
        scenario_id
    )

    if not scenario:
        return False

    scenario_id = (
        str(scenario_id)
        .upper()
        .strip()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"MENJALANKAN DATASET - "
        f"SKENARIO {scenario_id}"
    )

    print(
        "=" * 70
    )

    print(
        f"Chunk Size : "
        f"{scenario['chunk_size']}"
    )

    print(
        f"Overlap    : "
        f"{scenario['overlap']}"
    )

    print(
        f"Top-K      : "
        f"{scenario['top_k']}"
    )

    print(
        f"Keterangan : "
        f"{scenario['desc']}"
    )

    print(
        f"Vectorstore: {DB_PATH}"
    )

    print(
        f"Log        : {LOG_FILE}"
    )

    print(
        "=" * 70
    )

    return run_dataset()


# ==============================
# 9. RUN DATASET
# ==============================

def run_dataset():

    """
    Menjalankan dataset otomatis
    tanpa menghitung metrik evaluasi.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        f"MEMULAI DATASET BOT "
        f"(Skenario: {SCENARIO_NAME})"
    )

    print(
        "=" * 60 + "\n"
    )

    if not DATASET_EVAL_FILE.exists():

        print(
            "[ERROR] Dataset tidak ditemukan: "
            f"{DATASET_EVAL_FILE}"
        )

        return

    try:

        with open(
            DATASET_EVAL_FILE,
            "r",
            encoding="utf-8-sig"
        ) as f:

            dataset = json.load(f)

    except Exception as e:

        print(
            f"[ERROR] Gagal memuat dataset: "
            f"{e}"
        )

        return

    if not dataset:

        print(
            "[ERROR] Dataset kosong."
        )

        return

    print(
        f"[OK] Dataset berhasil dimuat: "
        f"{len(dataset)} pertanyaan"
    )

    print(
        "=" * 60 + "\n"
    )

    if os.path.exists(LOG_FILE):

        os.remove(LOG_FILE)

        print(
            "[INFO] Log lama dihapus.\n"
        )

    conversation_history.clear()

    success_count = 0
    skip_count = 0

    for i, item in enumerate(
        dataset,
        1
    ):

        query = (
            item.get(
                "question",
                item.get(
                    "user_input",
                    ""
                )
            )
            .strip()
        )

        ground_truth = (
            item.get(
                "ground_truth",
                item.get(
                    "reference",
                    ""
                )
            )
            .strip()
        )

        if not query:

            skip_count += 1

            continue

        print(
            f"[{i}/{len(dataset)}] "
            f"Pertanyaan: "
            f"{query[:70]}..."
        )

        if detect_intent(
            query
        )["is_greeting"]:

            print(
                "   [SKIP] Greeting detected\n"
            )

            skip_count += 1

            continue

        try:

            generate_response(
                query,
                ground_truth
            )

            success_count += 1

            print(
                "   [OK] Jawaban tersimpan.\n"
            )

        except Exception as e:

            print(
                f"   [ERROR] "
                f"{type(e).__name__}: {e}\n"
            )

    print(
        "=" * 60
    )

    print(
        f"[SELESAI] "
        f"{success_count} pertanyaan "
        "berhasil dijalankan."
    )

    print(
        f"[SKIP] "
        f"{skip_count} pertanyaan dilewati."
    )

    print(
        f"[LOG] {LOG_FILE}"
    )

    print(
        "=" * 60 + "\n"
    )


def run_all_scenarios():

    """
    Menjalankan seluruh skenario
    S1 sampai S9 secara berurutan.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "MEMULAI SEMUA SKENARIO RAG"
    )

    print(
        "=" * 70
    )

    results = {}

    for scenario_id in (
        get_all_scenarios().keys()
    ):

        results[
            scenario_id
        ] = run_single_scenario(
            scenario_id
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "RINGKASAN SEMUA SKENARIO"
    )

    print(
        "=" * 70
    )

    for scenario_id, success in (
        results.items()
    ):

        status = (
            "BERHASIL"
            if success
            else "GAGAL"
        )

        print(
            f"{scenario_id}: {status}"
        )

    print(
        "=" * 70
    )


# ==============================
# 10. MODE EVALUASI
# ==============================

def run_evaluation():

    """
    Menjalankan chatbot otomatis
    menggunakan dataset evaluasi.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        f"MEMULAI MODE EVALUASI RAG "
        f"(Skenario: {SCENARIO_NAME})"
    )

    print(
        "=" * 60
    )

    if not DATASET_EVAL_FILE.exists():

        print(
            "[ERROR] File dataset evaluasi "
            f"tidak ditemukan: "
            f"{DATASET_EVAL_FILE}"
        )

        return

    try:

        with open(
            DATASET_EVAL_FILE,
            "r",
            encoding="utf-8-sig"
        ) as f:

            dataset = json.load(f)

    except Exception as e:

        print(
            f"[ERROR] Gagal memuat dataset: "
            f"{e}"
        )

        return

    if not dataset:

        print(
            "[ERROR] Dataset evaluasi kosong."
        )

        return

    print(
        f"[OK] Dataset berhasil dimuat: "
        f"{len(dataset)} pertanyaan"
    )

    print(
        "=" * 60 + "\n"
    )

    if os.path.exists(LOG_FILE):

        os.remove(LOG_FILE)

        print(
            "[INFO] Log lama dihapus "
            "untuk evaluasi baru.\n"
        )

    conversation_history.clear()

    evaluated_count = 0
    skipped_count = 0

    for i, item in enumerate(
        dataset,
        1
    ):

        query = (
            item.get(
                "question",
                item.get(
                    "user_input",
                    ""
                )
            )
            .strip()
        )

        ground_truth = (
            item.get(
                "ground_truth",
                item.get(
                    "reference",
                    ""
                )
            )
            .strip()
        )

        if not query:

            skipped_count += 1

            continue

        print(
            f"[{i}/{len(dataset)}] "
            f"Pertanyaan: "
            f"{query[:70]}..."
        )

        intents = detect_intent(
            query
        )

        if intents["is_greeting"]:

            print(
                "   [SKIP] Greeting detected"
            )

            skipped_count += 1

            continue

        try:

            generate_response(
                query,
                ground_truth
            )

            evaluated_count += 1

            print(
                "   [OK] Jawaban & ground_truth "
                "tersimpan.\n"
            )

        except Exception as e:

            print(
                f"   [ERROR] "
                f"{type(e).__name__}: {e}\n"
            )

    print(
        "=" * 60
    )

    print(
        f"[SELESAI] "
        f"{evaluated_count} pertanyaan "
        "telah diuji."
    )

    print(
        f"[SKIP] "
        f"{skipped_count} pertanyaan dilewati."
    )

    print(
        "=" * 60 + "\n"
    )


# ==============================
# 11. MAIN LOOP
# ==============================

def main():

    """
    Entry point aplikasi chatbot.
    """

    print(
        "=" * 60
    )

    print(
        "CHATBOT EDUKASI KESEHATAN "
        "IBU DAN ANAK"
    )

    print(
        f"Skenario Aktif: "
        f"{SCENARIO_NAME}"
    )

    print(
        "=" * 60
    )

    print(
        "Perintah khusus:"
    )

    print(
        "  /exit  -> Keluar dari chatbot"
    )

    print(
        "  /logs  -> Lihat riwayat interaksi"
    )

    print(
        "  /help  -> Tampilkan panduan"
    )

    print(
        "-" * 60
    )

    print(
        "\nMode lain:"
    )

    print(
        "python .\\app\\chatbot.py convert"
    )

    print(
        "python .\\app\\chatbot.py eval"
    )

    print(
        "python .\\app\\chatbot.py "
        "eval-scenario S5"
    )

    print(
        "python .\\app\\chatbot.py "
        "eval-all"
    )

    print(
        "python .\\app\\chatbot.py chat"
    )

    print(
        "-" * 60 + "\n"
    )

    while True:

        try:

            query = input(
                "Anda: "
            ).strip()

            if not query:

                print(
                    "Chatbot: Silakan masukkan "
                    "pertanyaan terlebih dahulu."
                )

                continue

            if query.lower() in [
                "/exit",
                "exit",
                "keluar",
                "selesai",
            ]:

                print(
                    "\nTerima kasih telah "
                    "menggunakan chatbot ini!"
                )

                print(
                    "Disclaimer: Chatbot ini "
                    "bukan pengganti konsultasi "
                    "medis profesional."
                )

                break

            elif query.lower() in [
                "/logs",
                "logs",
                "riwayat",
            ]:

                show_logs(10)

                continue

            elif query.lower() in [
                "/help",
                "help",
                "?",
            ]:

                print(
                    "\nPANDUAN PENGGUNAAN:"
                )

                print(
                    "- Ketik pertanyaan "
                    "kesehatan ibu/anak "
                    "secara alami"
                )

                print(
                    "- Contoh: "
                    "'Apa jadwal imunisasi "
                    "bayi 6 bulan?'"
                )

                print(
                    "- Untuk kondisi darurat, "
                    "hubungi tenaga kesehatan."
                )

                continue

            if query.lower() in [
                "lanjutkan",
                "lanjut",
                "continue",
            ]:

                print(
                    "Chatbot: ",
                    end="",
                    flush=True
                )

                answer, was_truncated = (
                    continue_response()
                )

                print(answer)

                if was_truncated:

                    print(
                        "\nTips: Jawaban masih "
                        "terpotong. Ketik "
                        "'lanjutkan' lagi jika perlu."
                    )

                print(
                    "\n" + "-" * 60
                )

                continue

            print(
                "Chatbot: ",
                end="",
                flush=True
            )

            answer, was_truncated = (
                generate_response(
                    query
                )
            )

            print(answer)

            if was_truncated:

                print(
                    "\nTips: Jawaban terpotong. "
                    "Ketik 'lanjutkan' "
                    "untuk melanjutkan."
                )

            print(
                "\n" + "-" * 60
            )

        except KeyboardInterrupt:

            print(
                "\n\nInterupsi terdeteksi. "
                "Keluar dengan aman..."
            )

            break

        except Exception as e:

            print(
                f"\n[Error] "
                f"{type(e).__name__}: {e}"
            )

            print(
                "Solusi: Periksa koneksi "
                "LM Studio atau restart aplikasi.\n"
            )


# ==============================
# 12. ENTRY POINT
# ==============================

if __name__ == "__main__":

    if len(sys.argv) > 1:

        mode = (
            sys.argv[1]
            .strip()
            .lower()
        )

        if mode == "convert":

            convert_csv_to_json()

        elif mode == "eval":

            run_evaluation()

        elif mode == "eval-scenario":

            if len(sys.argv) < 3:

                print(
                    "[ERROR] Masukkan ID skenario."
                )

                print(
                    "Contoh: "
                    "python .\\app\\chatbot.py "
                    "eval-scenario S5"
                )

            else:

                run_single_scenario(
                    sys.argv[2]
                )

        elif mode == "run-dataset":

            run_dataset()

        elif mode == "run-scenario":

            if len(sys.argv) < 3:

                print(
                    "[ERROR] Masukkan ID skenario."
                )

                print(
                    "Contoh: "
                    "python .\\app\\chatbot.py "
                    "run-scenario S5"
                )

            else:

                run_dataset_scenario(
                    sys.argv[2]
                )

        elif mode == "eval-all":

            run_all_scenarios()

        elif mode == "chat":

            main()

        else:

            print(
                f"[ERROR] Mode tidak dikenali: "
                f"{sys.argv[1]}"
            )

            print(
                "Gunakan:"
            )

            print(
                "python .\\app\\chatbot.py convert"
            )

            print(
                "python .\\app\\chatbot.py eval"
            )

            print(
                "python .\\app\\chatbot.py "
                "eval-scenario S5"
            )

            print(
                "python .\\app\\chatbot.py eval-all"
            )

            print(
                "python .\\app\\chatbot.py chat"
            )

    else:

        main()