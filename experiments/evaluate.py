import argparse
import json
import os
import warnings
from pathlib import Path
from urllib.parse import urlparse
warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*is being sunset.*"
)
warnings.filterwarnings(
    "ignore",
    message="The class `HuggingFaceEmbeddings` was deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    message="LangchainEmbeddingsWrapper is deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    message="Importing .* from 'ragas.metrics' is deprecated.*",
    category=DeprecationWarning,
)
import pandas as pd
from datasets import Dataset
from langchain_community.embeddings import (
    HuggingFaceEmbeddings as LangchainHuggingFaceEmbeddings
)
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.run_config import RunConfig
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from config.config import (
    PROJECT_ROOT,
    SCENARIO_NAME,
    TOP_K,
)
# =========================================================
# KONFIGURASI
# =========================================================
LOG_FILE = (
    PROJECT_ROOT
    / "results"
    / "chatbot_logs.jsonl"
)
DEFAULT_DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "questions"
    / "eval_dataset.jsonl"
)
EMBEDDING_MODEL_PATH = str(
    PROJECT_ROOT
    / "models"
    / "bge-m3"
)
LLM_API_URL = "http://127.0.0.1:1234/v1"
LLM_MODEL = "google/gemma-4-e2b"
# =========================================================
# NORMALISASI DATA
# =========================================================
def _as_text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]
    value = str(value).strip()
    return [value] if value else []
def _reference_from_entry(entry: dict) -> str:
    for key in (
        "reference",
        "ground_truth",
        "ground_truths",
    ):
        value = entry.get(key)
        if isinstance(value, list):
            value = "\n\n".join(
                _as_text_list(value)
            )
        if value and str(value).strip():
            return str(value).strip()
    return ""
def _truncate_text(
    value: str,
    max_chars: int
) -> str:
    if len(value) <= max_chars:
        return value
    cut = value[:max_chars]
    boundary = cut.rfind(".")
    if boundary > max_chars * 0.7:
        return cut[:boundary + 1]
    return cut
def _limit_contexts(
    contexts: list[str],
    max_chars: int
) -> list[str]:
    limited = []
    remaining = max_chars
    for context in contexts:
        if remaining <= 0:
            break
        excerpt = _truncate_text(
            context,
            remaining
        )
        if excerpt:
            limited.append(excerpt)
            remaining -= len(excerpt)
    return limited
# =========================================================
# MEMBACA SATU RECORD
# =========================================================
def _record_from_entry(
    entry: dict,
    max_context_chars: int
) -> dict | None:
    # -----------------------------------------------------
    # USER INPUT
    # -----------------------------------------------------
    user_input = (
        entry.get("user_input")
        or entry.get("question")
        or entry.get("query")
    )
    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------
    response = (
        entry.get("response")
        or entry.get("answer")
    )
    if not user_input or not response:
        return None
    # -----------------------------------------------------
    # CONTEXT
    # -----------------------------------------------------
    raw_contexts = entry.get(
        "retrieved_contexts"
    )
    if raw_contexts is None:
        raw_contexts = entry.get(
            "contexts"
        )
    if raw_contexts is None:
        raw_contexts = entry.get(
            "context",
            []
        )
    retrieved_contexts = _as_text_list(
        raw_contexts
    )
    retrieved_contexts = _limit_contexts(
        retrieved_contexts,
        max_context_chars
    )
    # -----------------------------------------------------
    # DETEKSI GUARDRAIL
    # -----------------------------------------------------
    metadata = entry.get(
        "metadata",
        {}
    ) or {}
    guardrail_active = bool(
        metadata.get(
            "guardrail_active",
            False
        )
    )
    context_value = str(
        entry.get(
            "context",
            ""
        )
    ).strip().upper()
    if (
        guardrail_active
        or context_value == "GUARDRAIL"
    ):
        record_type = "guardrail"
    else:
        record_type = "rag"
    # -----------------------------------------------------
    # VALIDASI
    # -----------------------------------------------------
    if (
        record_type == "rag"
        and not retrieved_contexts
    ):
        return None
    # -----------------------------------------------------
    # RECORD FINAL
    # -----------------------------------------------------
    return {
        "interaction_id": entry.get(
            "interaction_id",
            ""
        ),
        "scenario": entry.get(
            "scenario",
            SCENARIO_NAME
        ),
        "rag_config": entry.get("rag_config", {}),
        "record_type": record_type,
        "user_input": str(
            user_input
        ).strip(),
        "response": str(
            response
        ).strip(),
        "retrieved_contexts":
            retrieved_contexts,
        "reference":
            _reference_from_entry(
                entry
            ),
    }
# =========================================================
# LOAD JSONL
# =========================================================
def load_jsonl_records(
    file_path: Path,
    limit: int | None = None,
    max_context_chars: int = 1800
) -> list[dict]:
    if not file_path.exists():
        print(
            f"[Error] File tidak ditemukan: "
            f"{file_path}"
        )
        return []
    records = []
    skipped_records = []
    print(
        f"Membaca data evaluasi dari: "
        f"{file_path}"
    )
    with file_path.open(
        "r",
        encoding="utf-8-sig"
    ) as f:
        for line_number, line in enumerate(
            f,
            1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                record = _record_from_entry(
                    entry,
                    max_context_chars
                )
                if record:
                    records.append(
                        record
                    )
                else:
                    skipped_records.append(
                        line_number
                    )
            except Exception as exc:
                skipped_records.append(
                    line_number
                )
                print(
                    f"[Warning] Baris "
                    f"{line_number} dilewati: "
                    f"{exc}"
                )
    if limit:
        records = records[:limit]
        print(
            f"Evaluasi dibatasi ke "
            f"{limit} record pertama."
        )
    print(
        f"Total record valid: "
        f"{len(records)}"
    )
    if skipped_records:
        print(
            f"Total record dilewati: "
            f"{len(skipped_records)}"
        )
        print(
            "Nomor baris yang dilewati: "
            f"{skipped_records}"
        )
    rag_count = sum(
        record["record_type"] == "rag"
        for record in records
    )
    guardrail_count = sum(
        record["record_type"] == "guardrail"
        for record in records
    )
    print(
        f"RAG records: {rag_count}"
    )
    print(
        f"Guardrail records: "
        f"{guardrail_count}"
    )
    return records
# =========================================================
# RAGAS LLM JUDGE
# =========================================================
def normalize_base_url(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(
            "LLM base URL kosong."
        )
    if not value.startswith(
        ("http://", "https://")
    ):
        value = (
            "http://" + value.lstrip("/")
        )
    value = value.rstrip("/")
    parsed = urlparse(value)
    if (
        parsed.scheme not in
        ("http", "https")
        or not parsed.netloc
    ):
        raise ValueError(
            "LLM base URL tidak valid: "
            f"{value}"
        )
    return value
def build_ragas_llm(args):
    base_url = normalize_base_url(
        args.llm_base_url
    )
    print(
        "Menghubungkan Ragas judge ke "
        f"LM Studio: {base_url} "
        f"({args.llm_model})"
    )
    openai_client = OpenAI(
        base_url=base_url,
        api_key=args.llm_api_key,
        max_retries=0,
        timeout=args.llm_timeout,
    )
    import instructor
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
# =========================================================
# EMBEDDING
# =========================================================
def build_ragas_embeddings():
    print(
        "Memuat embedding evaluator dari: "
        f"{EMBEDDING_MODEL_PATH}"
    )
    embeddings = (
        LangchainHuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )
    )
    return LangchainEmbeddingsWrapper(
        embeddings
    )
# =========================================================
# EVALUASI RAG
# =========================================================
def evaluate_rag_records(
    records: list[dict],
    args
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    print()
    print("=" * 60)
    print("EVALUASI RAG")
    print("=" * 60)
    print(
        f"Jumlah record RAG: "
        f"{len(records)}"
    )
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]
    print(
        "Metrik: "
        + ", ".join(
            metric.name
            for metric in metrics
        )
    )
    reference_count = sum(
        bool(
            record.get(
                "reference",
                ""
            ).strip()
        )
        for record in records
    )
    print(
        f"Reference valid: "
        f"{reference_count}/"
        f"{len(records)}"
    )
    dataset = Dataset.from_pandas(
        pd.DataFrame(records),
        preserve_index=False
    )
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=build_ragas_llm(
            args
        ),
        embeddings=(
            build_ragas_embeddings()
        ),
        run_config=RunConfig(
            timeout=int(
                args.llm_timeout
            ),
            max_retries=args.max_retries,
            max_workers=args.max_workers,
        ),
        raise_exceptions=args.debug,
    )
    return results.to_pandas()
# =========================================================
# EVALUASI GUARDRAIL
# =========================================================
def evaluate_guardrail_records(
    records: list[dict],
    args
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    print()
    print("=" * 60)
    print("EVALUASI GUARDRAIL")
    print("=" * 60)
    print(
        f"Jumlah record guardrail: "
        f"{len(records)}"
    )
    print(
        "Metrik: answer_relevancy"
    )
    dataset = Dataset.from_pandas(
        pd.DataFrame(records),
        preserve_index=False
    )
    results = evaluate(
        dataset=dataset,
        metrics=[
            answer_relevancy
        ],
        llm=build_ragas_llm(
            args
        ),
        embeddings=(
            build_ragas_embeddings()
        ),
        run_config=RunConfig(
            timeout=int(
                args.llm_timeout
            ),
            max_retries=args.max_retries,
            max_workers=args.max_workers,
        ),
        raise_exceptions=args.debug,
    )
    return results.to_pandas()
# =========================================================
# MEMPERTAHANKAN INDEX ASLI
# =========================================================
def _restore_original_indices(
    result_df: pd.DataFrame,
    source_records: list[dict],
) -> pd.DataFrame:
    """
    Memastikan hasil RAGAS kembali mempunyai
    _record_index asli.
    Prioritas:
    1. interaction_id
    2. _record_index
    3. posisi record sumber
    """
    if result_df is None or result_df.empty:
        return pd.DataFrame()
    result_df = result_df.copy()
    source_df = pd.DataFrame(
        source_records
    ).copy()
    # -----------------------------------------------------
    # PRIORITAS 1: interaction_id
    # -----------------------------------------------------
    if (
        "interaction_id" in result_df.columns
        and "interaction_id" in source_df.columns
    ):
        mapping = {}
        for _, row in source_df.iterrows():
            interaction_id = str(
                row.get(
                    "interaction_id",
                    ""
                )
            ).strip()
            if interaction_id:
                mapping[
                    interaction_id
                ] = row[
                    "_record_index"
                ]
        result_df["_record_index"] = (
            result_df["interaction_id"]
            .astype(str)
            .str.strip()
            .map(mapping)
        )
    # -----------------------------------------------------
    # PRIORITAS 2: _record_index
    # -----------------------------------------------------
    elif "_record_index" in result_df.columns:
        result_df["_record_index"] = (
            pd.to_numeric(
                result_df["_record_index"],
                errors="coerce"
            )
        )
    # -----------------------------------------------------
    # PRIORITAS 3: FALLBACK URUTAN
    # -----------------------------------------------------
    else:
        if len(result_df) != len(
            source_records
        ):
            raise RuntimeError(
                "Jumlah hasil evaluasi tidak sama "
                "dengan jumlah record sumber, "
                "sehingga _record_index tidak dapat "
                "dipetakan dengan aman."
            )
        result_df["_record_index"] = [
            record["_record_index"]
            for record in source_records
        ]
    # -----------------------------------------------------
    # VALIDASI INDEX
    # -----------------------------------------------------
    if result_df[
        "_record_index"
    ].isna().any():
        missing_count = int(
            result_df[
                "_record_index"
            ]
            .isna()
            .sum()
        )
        raise RuntimeError(
            f"{missing_count} hasil evaluasi "
            "tidak dapat dipetakan kembali "
            "ke record asli."
        )
    result_df["_record_index"] = (
        result_df["_record_index"]
        .astype(int)
    )
    # -----------------------------------------------------
    # VALIDASI DUPLIKAT
    # -----------------------------------------------------
    if result_df[
        "_record_index"
    ].duplicated().any():
        raise RuntimeError(
            "Terdapat _record_index duplikat "
            "pada hasil evaluasi."
        )
    return result_df
# =========================================================
# GABUNGKAN HASIL
# =========================================================
def combine_results(
    records: list[dict],
    rag_results: pd.DataFrame,
    guardrail_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menggabungkan hasil evaluasi RAG
    dan guardrail.
    """
    original_df = pd.DataFrame(
        records
    ).copy()
    result_frames = []
    # -----------------------------------------------------
    # RAG RESULTS
    # -----------------------------------------------------
    if (
        rag_results is not None
        and not rag_results.empty
    ):
        rag_results = (
            _restore_original_indices(
                rag_results,
                [
                    record
                    for record in records
                    if record["record_type"]
                    == "rag"
                ],
            )
        )
        result_frames.append(
            rag_results
        )
    # -----------------------------------------------------
    # GUARDRAIL RESULTS
    # -----------------------------------------------------
    if (
        guardrail_results is not None
        and not guardrail_results.empty
    ):
        guardrail_results = (
            _restore_original_indices(
                guardrail_results,
                [
                    record
                    for record in records
                    if record["record_type"]
                    == "guardrail"
                ],
            )
        )
        result_frames.append(
            guardrail_results
        )
    if not result_frames:
        return pd.DataFrame()
    results = pd.concat(
        result_frames,
        ignore_index=True
    )
    # -----------------------------------------------------
    # VALIDASI DUPLIKAT
    # -----------------------------------------------------
    if results[
        "_record_index"
    ].duplicated().any():
        raise RuntimeError(
            "Hasil RAG dan guardrail mempunyai "
            "_record_index yang duplikat."
        )
    # -----------------------------------------------------
    # IDENTITAS ASLI
    # -----------------------------------------------------
    identity_columns = [
        "_record_index",
        "interaction_id",
        "scenario",
        "record_type",
        "user_input",
        "response",
        "reference",
    ]
    identity_df = original_df[
        [
            col
            for col in identity_columns
            if col in original_df.columns
        ]
    ].copy()
        # -----------------------------------------------------
    # AMBIL METRIK SAJA
    # -----------------------------------------------------
    metric_columns = [
        "_record_index",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    metric_columns = [
        col
        for col in metric_columns
        if col in results.columns
    ]
    metric_df = results[
        metric_columns
    ].copy()
    # -----------------------------------------------------
    # TAMBAHKAN RECORD TYPE DARI DATA SUMBER
    # -----------------------------------------------------
    type_df = original_df[
        [
            "_record_index",
            "record_type",
        ]
    ].copy()
    metric_df = metric_df.merge(
        type_df,
        on="_record_index",
        how="left",
        validate="one_to_one",
    )
    # -----------------------------------------------------
    # GUARDRAIL RELEVANCY
    # -----------------------------------------------------
    if "answer_relevancy" in metric_df.columns:
        metric_df[
            "guardrail_feedback_relevancy"
        ] = metric_df[
            "answer_relevancy"
        ].where(
            metric_df["record_type"] == "guardrail"
        )
    else:
        metric_df[
            "guardrail_feedback_relevancy"
        ] = pd.NA
    metric_df = metric_df.drop(
        columns=["record_type"]
    )
    # -----------------------------------------------------
    # MERGE DENGAN IDENTITAS
    # -----------------------------------------------------
    results = identity_df.merge(
        metric_df,
        on="_record_index",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    # -----------------------------------------------------
    # GUARDRAIL DETECTED
    # -----------------------------------------------------
    results[
        "guardrail_detected"
    ] = (
        results["record_type"]
        == "guardrail"
    )
    # -----------------------------------------------------
    # METRIK RETRIEVAL TIDAK BERLAKU
    # UNTUK GUARDRAIL
    # -----------------------------------------------------
    for column in [
        "faithfulness",
        "context_precision",
        "context_recall",
    ]:
        if column in results.columns:
            results.loc[
                results["record_type"]
                == "guardrail",
                column
            ] = pd.NA
    return results
# =========================================================
# SUMMARY TERMINAL
# =========================================================
def print_summary(
    results_df: pd.DataFrame,
    evaluated_scenario: str = "unknown",
    evaluated_config: dict | None = None
):
    if evaluated_config is None:
        evaluated_config = {}
    print()
    print("=" * 60)
    print("RAGAS EVALUATION COMPLETED")
    print("=" * 60)
    print(f"Skenario: {evaluated_scenario}")
    print(f"Chunk Size: {evaluated_config.get('chunk_size', '-')}")
    print(f"Overlap: {evaluated_config.get('overlap', '-')}")
    print(f"TOP_K: {evaluated_config.get('top_k', '-')}")
    print(f"Total pertanyaan: {len(results_df)}")
    if "record_type" in results_df.columns:
        counts = results_df["record_type"].value_counts()
        print("\nJumlah record:")
        print(counts)
    rag_df = results_df[results_df["record_type"] == "rag"]
    print(f"\nRata-rata RAGAS ({len(rag_df)} record RAG):")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric not in rag_df.columns:
            continue
        values = pd.to_numeric(rag_df[metric], errors="coerce").dropna()
        print(f"  - {metric}: {values.mean():.4f}" if len(values) else f"  - {metric}: N/A")
    guardrail_df = results_df[results_df["record_type"] == "guardrail"]
    if not guardrail_df.empty:
        feedback_values = pd.to_numeric(guardrail_df["guardrail_feedback_relevancy"], errors="coerce").dropna()
        print("\nEvaluasi Guardrail:")
        print(f"  - Record guardrail: {len(guardrail_df)}")
        print(f"  - Guardrail terdeteksi/logged: {int(guardrail_df['guardrail_detected'].sum())}")
        print(f"  - Rata-rata relevansi feedback: {feedback_values.mean():.4f}" if len(feedback_values) else "  - Rata-rata relevansi feedback: N/A")
    print("=" * 60)
# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluasi chatbot RAG "
            "menggunakan Ragas, "
            "BGE-M3, dan LM Studio."
        )
    )
    parser.add_argument(
        "--source",
        choices=[
            "logs",
            "dataset"
        ],
        default="logs",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=int(
            os.getenv(
                "RAGAS_MAX_CONTEXT_CHARS",
                "1800"
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "ragas_evaluation_results.csv"
        ),
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv(
            "RAGAS_LLM_BASE_URL",
            LLM_API_URL
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv(
            "RAGAS_LLM_MODEL",
            LLM_MODEL
        ),
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.getenv(
            "RAGAS_LLM_API_KEY",
            "lm-studio"
        ),
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=float(
            os.getenv(
                "RAGAS_LLM_TIMEOUT",
                "300"
            )
        ),
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=int(
            os.getenv(
                "RAGAS_JUDGE_MAX_TOKENS",
                "4096"
            )
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(
            os.getenv(
                "RAGAS_MAX_WORKERS",
                "1"
            )
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(
            os.getenv(
                "RAGAS_MAX_RETRIES",
                "1"
            )
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()
    # =====================================================
    # VALIDASI URL
    # =====================================================
    try:
        args.llm_base_url = (
            normalize_base_url(
                args.llm_base_url
            )
        )
    except ValueError as exc:
        parser.error(
            str(exc)
        )
    # =====================================================
    # INPUT
    # =====================================================
    input_path = (
        args.input
        or (
            LOG_FILE
            if args.source == "logs"
            else DEFAULT_DATASET_FILE
        )
    )
    if args.max_context_chars <= 0:
        parser.error(
            "--max-context-chars harus "
            "lebih besar dari 0"
        )
    # =====================================================
    # LOAD DATA
    # =====================================================
    records = load_jsonl_records(
        input_path,
        args.limit,
        args.max_context_chars
    )
    scenario_values = {str(record.get("scenario", "")).strip() for record in records if record.get("scenario")}
    if len(scenario_values) == 1:
        evaluated_scenario = next(iter(scenario_values))
    elif len(scenario_values) > 1:
        evaluated_scenario = "mixed"
    else:
        evaluated_scenario = "unknown"
    config_values = [record.get("rag_config") for record in records if isinstance(record.get("rag_config"), dict) and record.get("rag_config")]
    evaluated_config = config_values[0] if config_values else {}
    if not records:
        print(
            "Tidak ada record valid."
        )
        return
    # =====================================================
    # RECORD INDEX ASLI
    # =====================================================
    for index, record in enumerate(
        records
    ):
        record[
            "_record_index"
        ] = index
    # =====================================================
    # PISAH RAG & GUARDRAIL
    # =====================================================
    rag_records = [
        record
        for record in records
        if record[
            "record_type"
        ] == "rag"
    ]
    guardrail_records = [
        record
        for record in records
        if record[
            "record_type"
        ] == "guardrail"
    ]
    print()
    print("=" * 60)
    print("TOTAL DATA")
    print("=" * 60)
    print(
        f"Total: "
        f"{len(records)}"
    )
    print(
        f"RAG: "
        f"{len(rag_records)}"
    )
    print(
        f"Guardrail: "
        f"{len(guardrail_records)}"
    )
    # =====================================================
    # VALIDASI PEMBAGIAN DATA
    # =====================================================
    if (
        len(rag_records)
        + len(guardrail_records)
        != len(records)
    ):
        raise RuntimeError(
            "Pembagian RAG dan guardrail "
            "tidak sesuai dengan jumlah "
            "record."
        )
    # =====================================================
    # EVALUASI
    # =====================================================
    rag_results = pd.DataFrame()
    guardrail_results = pd.DataFrame()
    # -----------------------------------------------------
    # EVALUASI RAG
    # -----------------------------------------------------
    try:
        if rag_records:
            rag_results = (
                evaluate_rag_records(
                    rag_records,
                    args
                )
            )
    except Exception as exc:
        print(
            "\n[ERROR EVALUASI RAG]"
        )
        print(exc)
        if args.debug:
            raise
        return
    # -----------------------------------------------------
    # EVALUASI GUARDRAIL
    # -----------------------------------------------------
    try:
        if guardrail_records:
            guardrail_results = (
                evaluate_guardrail_records(
                    guardrail_records,
                    args
                )
            )
    except Exception as exc:
        print(
            "\n[ERROR EVALUASI GUARDRAIL]"
        )
        print(exc)
        if args.debug:
            raise
        return
    # =====================================================
    # GABUNGKAN HASIL
    # =====================================================
    results_df = combine_results(
        records,
        rag_results,
        guardrail_results
    )
    if results_df.empty:
        print(
            "Tidak ada hasil evaluasi."
        )
        return
    # =====================================================
    # VALIDASI FINAL JUMLAH DATA
    # =====================================================
    if len(results_df) != len(records):
        raise RuntimeError(
            "Hasil evaluasi tidak lengkap: "
            f"{len(results_df)}/"
            f"{len(records)} record."
        )
    # =====================================================
    # VALIDASI RECORD INDEX
    # =====================================================
    expected_indices = set(
        range(len(records))
    )
    actual_indices = set(
        results_df[
            "_record_index"
        ].tolist()
    )
    if actual_indices != expected_indices:
        missing_indices = (
            expected_indices
            - actual_indices
        )
        extra_indices = (
            actual_indices
            - expected_indices
        )
        raise RuntimeError(
            "Index record tidak lengkap.\n"
            f"Missing: {sorted(missing_indices)}\n"
            f"Extra: {sorted(extra_indices)}"
        )
    # =====================================================
    # VALIDASI JUMLAH RAG + GUARDRAIL
    # =====================================================
    result_rag_count = int(
        (
            results_df["record_type"]
            == "rag"
        ).sum()
    )
    result_guardrail_count = int(
        (
            results_df["record_type"]
            == "guardrail"
        ).sum()
    )
    if (
        result_rag_count
        != len(rag_records)
    ):
        raise RuntimeError(
            "Jumlah RAG hasil evaluasi "
            "tidak sesuai."
        )
    if (
        result_guardrail_count
        != len(guardrail_records)
    ):
        raise RuntimeError(
            "Jumlah guardrail hasil evaluasi "
            "tidak sesuai."
        )
    # =====================================================
    # URUTKAN KOLOM
    # =====================================================
    preferred_columns = [
        "_record_index",
        "interaction_id",
        "scenario",
        "record_type",
        "user_input",
        "response",
        "reference",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "guardrail_detected",
        "guardrail_feedback_relevancy",
    ]
    existing_columns = [
        col
        for col in preferred_columns
        if col in results_df.columns
    ]
    remaining_columns = [
        col
        for col in results_df.columns
        if col not in existing_columns
    ]
    results_df = results_df[
        existing_columns
        + remaining_columns
    ]
    # =====================================================
    # URUTKAN SESUAI URUTAN PERTANYAAN ASLI
    # =====================================================
    results_df = (
        results_df
        .sort_values(
            "_record_index"
        )
        .reset_index(
            drop=True
        )
    )
    # =====================================================
    # SUMMARY TERMINAL
    # =====================================================
    print_summary(
        results_df,
        evaluated_scenario=evaluated_scenario,
        evaluated_config=evaluated_config
    )
    # =====================================================
    # SIMPAN DETAIL
    # =====================================================
    output_path = args.output
    if not output_path.is_absolute():
        output_path = (
            PROJECT_ROOT
            / output_path
        )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )
    print()
    print(
        "[1] Hasil detail tersimpan:"
    )
    print(
        output_path
    )
    # =====================================================
    # SUMMARY CSV
    # =====================================================
    # -----------------------------------------------------
    # SUMMARY RAG
    # -----------------------------------------------------
    rag_summary_df = results_df[
        results_df["record_type"]
        == "rag"
    ]
    summary_rows = []
    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:
        if metric not in (
            rag_summary_df.columns
        ):
            continue
        valid_values = pd.to_numeric(
            rag_summary_df[metric],
            errors="coerce"
        ).dropna()
        if len(valid_values):
            average_score = round(
                float(
                    valid_values.mean()
                ),
                4
            )
        else:
            average_score = None
        summary_rows.append(
            {
                "metric": metric,
                "average_score":
                    average_score,
                "evaluated_records":
                    len(valid_values),
                "evaluation_scope":
                    "RAG",
            }
        )
    # -----------------------------------------------------
    # SUMMARY GUARDRAIL
    # -----------------------------------------------------
    guardrail_values = pd.to_numeric(
        results_df.loc[
            results_df["record_type"]
            == "guardrail",
            "guardrail_feedback_relevancy",
        ],
        errors="coerce"
    ).dropna()
    summary_rows.append(
        {
            "metric":
                "guardrail_feedback_relevancy",
            "average_score":
                (
                    round(
                        float(
                            guardrail_values.mean()
                        ),
                        4
                    )
                    if len(
                        guardrail_values
                    )
                    else None
                ),
            "evaluated_records":
                len(
                    guardrail_values
                ),
            "evaluation_scope":
                "Guardrail",
        }
    )
    # -----------------------------------------------------
    # JUMLAH DATA
    # -----------------------------------------------------
    summary_rows.extend(
        [
            {
                "metric":
                    "total_questions",
                "average_score":
                    len(results_df),
                "evaluated_records":
                    len(results_df),
                "evaluation_scope":
                    "ALL",
            },
            {
                "metric":
                    "rag_questions",
                "average_score":
                    len(rag_records),
                "evaluated_records":
                    len(rag_records),
                "evaluation_scope":
                    "RAG",
            },
            {
                "metric":
                    "guardrail_questions",
                "average_score":
                    len(guardrail_records),
                "evaluated_records":
                    len(guardrail_records),
                "evaluation_scope":
                    "Guardrail",
            },
            {
                "metric":
                    "guardrail_detected_logged",
                "average_score":
                    int(
                        results_df[
                            "guardrail_detected"
                        ].sum()
                    ),
                "evaluated_records":
                    len(guardrail_records),
                "evaluation_scope":
                    "Guardrail",
            },
        ]
    )
    summary_df = pd.DataFrame(
        summary_rows
    )
    summary_path = (
        output_path.with_name(
            f"{output_path.stem}"
            "_summary.csv"
        )
    )
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )
    print(
        "[2] Ringkasan tersimpan:"
    )
    print(
        summary_path
    )
# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()