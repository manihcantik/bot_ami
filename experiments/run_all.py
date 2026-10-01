import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.scenarios import get_all_scenarios


DEFAULT_SOURCE = PROJECT_ROOT / "data" / "questions" / "questions_all.csv"


def load_questions(source: Path, limit: int) -> list[dict]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        records = []
        for row in rows:
            question = (row.get("question") or "").strip()
            ground_truth = (row.get("ground_truth") or "").strip()
            if question and ground_truth:
                records.append({"question": question, "ground_truth": ground_truth})
            if len(records) == limit:
                break
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jalankan dataset pertanyaan pada semua skenario RAG."
    )
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results"
    )
    args = parser.parse_args()

    if args.questions <= 0:
        parser.error("--questions harus lebih besar dari 0")
    if not args.source.exists():
        parser.error(f"Dataset tidak ditemukan: {args.source}")

    records = load_questions(args.source, args.questions)
    if len(records) < args.questions:
        print(f"[Info] Hanya ditemukan {len(records)} pertanyaan valid.")
    if not records:
        raise SystemExit("Tidak ada pertanyaan valid untuk dijalankan.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Menjalankan {len(records)} pertanyaan pada {len(get_all_scenarios())} skenario.")

    for scenario_id, scenario in get_all_scenarios().items():
        scenario_dir = args.output_dir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = scenario_dir / "dataset_evaluasi.json"
        log_path = scenario_dir / "chatbot_logs.jsonl"
        dataset_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        environment = os.environ.copy()
        environment.update(
            {
                "RAG_SCENARIO": scenario_id,
                "RAG_TOP_K": str(scenario["top_k"]),
                "RAG_CHUNK_SIZE": str(scenario["chunk_size"]),
                "RAG_CHUNK_OVERLAP": str(scenario["overlap"]),
                "RAG_CHROMA_DB_DIR": str(PROJECT_ROOT / "vectorstore" / scenario_id),
                "RAG_DATASET_EVAL_FILE": str(dataset_path),
                "RAG_LOG_FILE": str(log_path),
            }
        )

        print(f"\n===== {scenario_id} =====")
        subprocess.run(
            [sys.executable, "-m", "app.chatbot", "eval"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )

    print(f"\nSelesai. Log tersimpan di: {args.output_dir}")


if __name__ == "__main__":
    main()