import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "customer_selected_columns.csv"
INCOMING = ROOT / "simulator_batches" / "incoming"


class BatchLoader:
    def __init__(self, data_file, batch_size):
        self.data = pd.read_csv(data_file, dtype={"CustomerID": "string"})
        self.batch_size = batch_size
        self.idx = 0

    def generate(self, batch_number):
        stop = min(self.idx + self.batch_size, len(self.data))
        batch = self.data.iloc[self.idx:stop]
        if batch.empty:
            raise ValueError("No more records are available.")

        INCOMING.mkdir(parents=True, exist_ok=True)
        path = INCOMING / f"batch_{batch_number:02d}.csv"
        batch.to_csv(path, index=False, encoding="utf-8-sig")
        self.idx = stop

        print(f"Created {path.relative_to(ROOT)} | customers: {len(batch)}")
        return path


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def push_batch(path):
    relative_path = path.relative_to(ROOT).as_posix()
    git("add", "--", relative_path)

    # Commit only this batch, even if other files are staged.
    if git("diff", "--cached", "--name-only", "--", relative_path):
        git("commit", "--only", "-m", f"Add {path.name}", "--", relative_path)

    git("push", "origin", "HEAD:refs/heads/data-automation")
    print(f"Pushed {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    if args.push and git("branch", "--show-current") != "data-automation":
        raise SystemExit("Switch to the data-automation branch before pushing.")

    loader = BatchLoader(SOURCE, batch_size=10)

    for number in (1, 2):
        batch_path = loader.generate(number)
        if args.push:
            push_batch(batch_path)
        if number == 1:
            time.sleep(3)
    