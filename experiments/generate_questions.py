"""Generate a large evaluation question set from all source-document chunks."""

import argparse
import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNK_DIR = PROJECT_ROOT / "data" / "chunks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "questions"


def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "")
    return value.strip()


def display_topic(item: dict) -> str:
    title = clean_text(item.get("title", ""))
    if title and title.lower() != "document":
        return title
    category = clean_text(item.get("category", "materi kesehatan"))
    sub_category = clean_text(item.get("sub_category", "")).replace("_", " ")
    return sub_category or category


def build_questions(item: dict, per_chunk: int) -> list[dict]:
    content = clean_text(item.get("content", ""))
    if len(content.split()) < 25:
        return []

    topic = display_topic(item)
    templates = [
        (
            f"Apa informasi penting tentang {topic} yang dijelaskan "
            "dalam materi ini?"
        ),
        f"Apa yang perlu diketahui ibu dan keluarga mengenai {topic}?",
        f"Apa anjuran atau langkah yang dijelaskan terkait {topic}?",
        (
            f"Apa hal yang perlu diperhatikan terkait {topic} "
            "berdasarkan materi ini?"
        ),
        (
            f"Bagaimana cara menerapkan informasi tentang {topic} "
            "dalam kehidupan sehari-hari?"
        ),
    ]

    source = clean_text(item.get("source", ""))
    records = []
    for question in templates[:per_chunk]:
        records.append(
            {
                "question": question,
                "ground_truth": content,
                "source": source,
                "title": topic,
                "category": clean_text(item.get("category", "")),
                "sub_category": clean_text(item.get("sub_category", "")),
                "chunk_id": str(item.get("chunk_id", "")),
                "question_type": "berbasis_materi",
            }
        )
    return records


def load_chunks(chunk_dir: Path) -> tuple[list[dict], list[Path]]:
    chunks = []
    files = []
    for path in sorted(chunk_dir.glob("*.json")):
        if path.name in {"experiment_summary.json", "comparison_results.json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARNING] File dilewati {path.name}: {exc}")
            continue
        if not isinstance(data, list):
            continue
        valid_items = [
            item for item in data
            if isinstance(item, dict) and item.get("content")
        ]
        if valid_items:
            files.append(path)
            chunks.extend(valid_items)
    return chunks, files


def write_outputs(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "questions_all.jsonl"
    json_path = output_dir / "questions_all.json"
    csv_path = output_dir / "questions_all.csv"

    jsonl_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        ),
        encoding="utf-8-sig",
    )
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "total_questions": len(records),
        "source_files": sorted({record["source"] for record in records}),
        "output_files": [jsonl_path.name, json_path.name, csv_path.name],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Buat pertanyaan evaluasi dari seluruh hasil chunking sumber data."
        )
    )
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--per-chunk",
        type=int,
        default=3,
        help="Jumlah pertanyaan per chunk (1-5, default: 3).",
    )
    args = parser.parse_args()
    if not 1 <= args.per_chunk <= 5:
        parser.error("--per-chunk harus berada di antara 1 dan 5")

    chunks, files = load_chunks(args.chunk_dir)
    records = []
    seen = set()
    for item in chunks:
        for record in build_questions(item, args.per_chunk):
            key = (record["question"], record["ground_truth"])
            if key not in seen:
                seen.add(key)
                records.append(record)

    if not records:
        raise SystemExit(
            "Tidak ada chunk valid untuk dibuat menjadi pertanyaan."
        )

    write_outputs(records, args.output_dir)
    print(f"[OK] {len(files)} file chunk dibaca.")
    print(f"[OK] {len(chunks)} chunk diproses.")
    print(f"[OK] {len(records)} pertanyaan tersimpan di: {args.output_dir}")


if __name__ == "__main__":
    main()
