from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
PLOTS = RESULTS /"plots"
DAGS = RESULTS / "dags"
SPAM_RAW = ROOT / "data" / "raw" / "spam"
SPAM_RAW_2 = ROOT / "data" / "raw" / "spam2" # if necessary, I personally only used the first spam dataset

def dataset_path(n):
    return DATA_RAW / f"gmail_dataset_{n}.jsonl"

def emails_to_move_path(n):
    return DATA_RAW / f"emails_to_move_{n}.csv"

def arviz_plots_path(plot_name, n):
    return PLOTS / f"{plot_name}_{n}.png"

def dags_path(plot_name, n):
    return DAGS / f"{plot_name}_{n}.png"