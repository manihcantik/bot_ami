import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.scenarios import get_all_scenarios, get_scenario


def run_command(command):
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def run_single_scenario(scenario_id):
    config = get_scenario(scenario_id)
    if config is None:
        raise ValueError(f"Skenario tidak ditemukan: {scenario_id}")

    chunk_dir = PROJECT_ROOT / "data" / "chunks" / scenario_id
    chunk_file = chunk_dir / f"kia_chunks_{scenario_id}.json"
    chroma_dir = PROJECT_ROOT / "vectorstore" / scenario_id
    chunk_dir.mkdir(parents=True, exist_ok=True)
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    print(f"\nMenjalankan {scenario_id}: {config['desc']}")
    run_command([
        sys.executable, "experiments/chunking.py",
        "--chunk-size", str(config["chunk_size"]),
        "--overlap", str(config["overlap"]),
        "--output", str(chunk_file),
    ])
    run_command([
        sys.executable, "experiments/embedding.py",
        "--input", str(chunk_file),
        "--chroma-path", str(chroma_dir),
        "--top-k", str(config["top_k"]),
        "--chunk-size", str(config["chunk_size"]),
        "--overlap", str(config["overlap"]),
        "--reset",
    ])
    return {
        "scenario": scenario_id,
        "config": config,
        "chunk_file": str(chunk_file),
        "chroma_dir": str(chroma_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Jalankan eksperimen RAG S1-S9")
    parser.add_argument("--scenario", choices=[*get_all_scenarios(), "all"], default="all")
    args = parser.parse_args()
    scenario_ids = list(get_all_scenarios()) if args.scenario == "all" else [args.scenario]
    results = {scenario_id: run_single_scenario(scenario_id) for scenario_id in scenario_ids}
    summary_path = PROJECT_ROOT / "results" / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ringkasan tersimpan di {summary_path}")


if __name__ == "__main__":
    main()