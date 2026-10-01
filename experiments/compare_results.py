import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Bandingkan statistik hasil eksperimen")
    parser.add_argument("--summary", default=str(PROJECT_ROOT / "results" / "experiment_summary.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "comparison.csv"))
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = []
    for scenario_id, data in summary.items():
        config = data["config"]
        chunk_path = Path(data["chunk_file"])
        chunks = json.loads(chunk_path.read_text(encoding="utf-8")) if chunk_path.exists() else []
        lengths = [len(item.get("content", "")) for item in chunks]
        rows.append({
            "Scenario": scenario_id,
            "Chunk Size": config["chunk_size"],
            "Overlap": config["overlap"],
            "Top-K": config["top_k"],
            "Description": config["desc"],
            "Total Chunks": len(chunks),
            "Avg Chunk Length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "Status": "OK" if chunks else "FAILED",
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else ["Scenario"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Hasil perbandingan tersimpan di {output_path}")


if __name__ == "__main__":
    main()