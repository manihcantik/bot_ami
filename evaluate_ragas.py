import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="`langchain-community` is being sunset.*")
warnings.filterwarnings("ignore", message="The class `HuggingFaceEmbeddings` was deprecated.*")
warnings.filterwarnings("ignore", message="LangchainEmbeddingsWrapper is deprecated.*")
warnings.filterwarnings(
    "ignore",
    message="Importing .* from 'ragas.metrics' is deprecated.*",
    category=DeprecationWarning,
)

import pandas as pd
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings as LangchainHuggingFaceEmbeddings
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.llms.base import InstructorLLM, InstructorModelArgs

from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config import PROJECT_ROOT, SCENARIO_NAME, TOP_K


LOG_FILE = PROJECT_ROOT / "chatbot_logs.jsonl"
DEFAULT_DATASET_FILE = PROJECT_ROOT / "eval_dataset.jsonl"
EMBEDDING_MODEL_PATH = str(PROJECT_ROOT / "bge-m3")
LLM_API_URL = "http://127.0.0.1:1234/v1"
LLM_MODEL = "google/gemma-4-e2b"


def _as_text_list(value) -> list[str]:
    """Normalize contexts/reference values from logs or jsonl files."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    value = str(value).strip()
    return [value] if value else []


def _reference_from_entry(entry: dict) -> str:
    """Support both old and new log field names."""
    for key in ("reference", "ground_truth", "ground_truths"):
        value = entry.get(key)
        if isinstance(value, list):
            value = "\n\n".join(_as_text_list(value))
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _record_from_entry(entry: dict) -> dict | None:
    metadata = entry.get("metadata", {}) or {}
    retrieved_contexts = _as_text_list(
        entry.get("retrieved_contexts", entry.get("contexts", []))
    )

    if metadata.get("fallback") or not retrieved_contexts:
        return None

    user_input = entry.get("user_input") or entry.get("question") or entry.get("query")
    response = entry.get("response") or entry.get("answer")
    if not user_input or not response:
        return None

    return {
        "user_input": str(user_input).strip(),
        "response": str(response).strip(),
        "retrieved_contexts": retrieved_contexts,
        "reference": _reference_from_entry(entry),
    }


def load_jsonl_records(file_path: Path, limit: int | None = None) -> list[dict]:
    if not file_path.exists():
        print(f"[Error] File tidak ditemukan: {file_path}")
        return []

    records = []
    print(f"Membaca data evaluasi dari: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                record = _record_from_entry(entry)
                if record:
                    records.append(record)
            except Exception as exc:
                print(f"[Warning] Baris {line_number} dilewati: {exc}")

    if limit:
        records = records[:limit]
        print(f"Evaluasi dibatasi ke {limit} record pertama.")

    print(f"Total record valid: {len(records)}")
    return records


def choose_metrics(records: list[dict], include_reference_metrics: bool):
    metrics = [faithfulness, answer_relevancy]
    has_reference = all(record.get("reference") for record in records)

    if include_reference_metrics and has_reference:
        metrics.extend([context_recall, context_precision])
    elif include_reference_metrics:
        print(
            "[Info] context_recall/context_precision dilewati karena ada record "
            "tanpa reference/ground_truth."
        )

    return metrics

def build_ragas_llm(args):
    print(
        f"Menghubungkan Ragas judge ke LM Studio: "
        f"{args.llm_base_url} ({args.llm_model})"
    )

    openai_client = OpenAI(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        max_retries=0,
        timeout=args.llm_timeout,
    )

    import instructor

    # Gunakan JSON Schema agar output sesuai format yang diharapkan RAGAS
    patched_client = instructor.from_openai(
        openai_client,
        mode=instructor.Mode.JSON_SCHEMA,
    )

    return InstructorLLM(
        client=patched_client,
        model=args.llm_model,
        provider="openai",
        model_args=InstructorModelArgs(
            temperature=0,
            top_p=1.0,
            max_tokens=args.judge_max_tokens,
        ),
    )

def build_ragas_embeddings():
    print(f"Memuat embedding evaluator dari: {EMBEDDING_MODEL_PATH}")
    embeddings = LangchainHuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(embeddings)


def print_summary(results_df: pd.DataFrame):
    metric_columns = [
        col
        for col in (
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
        )
        if col in results_df.columns
    ]

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION COMPLETED")
    print("=" * 60)
    print(f"Skenario: {SCENARIO_NAME} | TOP_K: {TOP_K}")

    print("\nRata-rata skor:")
    for col in metric_columns:
        print(f"  - {col}: {results_df[col].mean():.4f}")

    print("\nDetail singkat per pertanyaan:")
    preview_columns = ["user_input", *metric_columns]
    print(results_df[preview_columns].to_string(index=False))

    if metric_columns and results_df[metric_columns].isna().all().all():
        print("\n[Diagnosis]")
        print(
            "Semua skor NaN. Biasanya ini berarti LLM judge Ragas gagal "
            "mengembalikan JSON/structured output yang valid."
        )
        print("Coba jalankan ulang dengan:")
        print("  python evaluate_ragas.py --source logs --limit 1 --debug")
        print("Jika error terkait parsing JSON, gunakan model judge yang lebih patuh JSON/structured output.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi chatbot RAG lokal memakai Ragas, BGE-M3, dan LM Studio."
    )
    parser.add_argument(
        "--source",
        choices=["logs", "dataset"],
        default="logs",
        help="logs = chatbot_logs.jsonl, dataset = eval_dataset.jsonl",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path JSONL custom. Default mengikuti --source.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah record")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "ragas_evaluation_results.csv",
        help="Path output CSV",
    )
    parser.add_argument(
        "--no-reference-metrics",
        action="store_true",
        help="Evaluasi hanya faithfulness dan answer_relevancy.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv("RAGAS_LLM_BASE_URL", LLM_API_URL),
        help="Base URL OpenAI-compatible untuk LLM judge.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("RAGAS_LLM_MODEL", LLM_MODEL),
        help="Nama model judge di LM Studio.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.getenv("RAGAS_LLM_API_KEY", "lm-studio"),
        help="API key dummy/real untuk endpoint OpenAI-compatible.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=float(os.getenv("RAGAS_LLM_TIMEOUT", "300")),  # UBAH DARI 180 KE 300
        help="Timeout request LLM judge dalam detik.",
    )
    parser.add_argument(
        "--adapter",
        choices=["auto", "instructor"],
        default=os.getenv("RAGAS_LLM_ADAPTER", "auto"),
        help="Adapter structured output Ragas jika --instructor-mode=json.",
    )
    parser.add_argument(
        "--instructor-mode",
        choices=["json_schema", "json"],
        default=os.getenv("RAGAS_INSTRUCTOR_MODE", "json_schema"),
        help="Mode structured output. LM Studio umumnya butuh json_schema.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=int(os.getenv("RAGAS_JUDGE_MAX_TOKENS", "4096")),
        help="Maksimum token respons LLM judge.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tampilkan error asli Ragas, bukan mengubah error menjadi NaN.",
    )
    args = parser.parse_args()

    input_path = args.input or (LOG_FILE if args.source == "logs" else DEFAULT_DATASET_FILE)
    records = load_jsonl_records(input_path, args.limit)
    if not records:
        print(
            "Belum ada data valid. Jalankan chatbot dulu, atau isi eval_dataset.jsonl "
            "dengan user_input, response, retrieved_contexts, dan reference."
        )
        return

    metrics = choose_metrics(records, include_reference_metrics=not args.no_reference_metrics)
    print("Metrik aktif: " + ", ".join(metric.name for metric in metrics))

    dataset = Dataset.from_pandas(pd.DataFrame(records), preserve_index=False)

    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=build_ragas_llm(args),
            embeddings=build_ragas_embeddings(),
            raise_exceptions=args.debug,
        )
    except Exception as exc:
        print(f"\n[Error evaluasi] {exc}")
        print("Pastikan LM Studio berjalan di http://127.0.0.1:1234 dan model sudah loaded.")
        return

    results_df = results.to_pandas()
    print_summary(results_df)

    # ==========================================
    # BAGIAN PENYIMPANAN HASIL (DIPERBARUI)
    # ==========================================
    
    # 1. Simpan Detail Evaluasi per Pertanyaan
    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[1] Hasil detail per pertanyaan tersimpan di: {output_path}")

    # 2. Simpan Rata-rata Skor Khusus 4 Metrik RAGAS (File Ringkasan)
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary_rows = []
    
    for metric in metric_names:
        if metric in results_df.columns:
            avg_score = results_df[metric].mean()
            # Jika hasilnya NaN (gagal dinilai), kita biarkan None/NaN
            score_val = round(float(avg_score), 4) if not pd.isna(avg_score) else None
            summary_rows.append({"metric": metric, "average_score": score_val})
            
    summary_df = pd.DataFrame(summary_rows)
    summary_path = PROJECT_ROOT / "ragas_4_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[2] Ringkasan rata-rata 4 metrik RAGAS tersimpan di: {summary_path}")


if __name__ == "__main__":
    main()