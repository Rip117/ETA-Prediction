import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

RAILPULL_DIR = "railpull"
DELAYS_JSON = os.path.join(RAILPULL_DIR, "data", "out", "delays.json")
HISTORY_CSV = os.path.join("data", "delay_history.csv")
INTERVAL_SECONDS = 15 * 60   # one snapshot every 15 minutes


def take_snapshot():
    # run railpull's poller once (it must run from inside the railpull folder)
    subprocess.run(
        [sys.executable, os.path.join("ntes", "poll_delays.py")],
        cwd=RAILPULL_DIR,
        check=True,
    )

    with open(DELAYS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    snapshot_time = datetime.now().isoformat(timespec="seconds")
    rows = []
    for train_no, info in data.get("trains", {}).items():
        rows.append([
            snapshot_time,
            train_no,
            info.get("d", 0),   # delay in minutes
            info.get("c", 0),   # 1 = cancelled
        ])
    return rows


def append_rows(rows):
    os.makedirs(os.path.dirname(HISTORY_CSV), exist_ok=True)
    new_file = not os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["snapshot_time", "train_no", "delay_min", "cancelled"])
        writer.writerows(rows)


if __name__ == "__main__":
    while True:
        try:
            rows = take_snapshot()
            append_rows(rows)
            print(f"{datetime.now():%H:%M:%S} saved {len(rows)} rows")
        except Exception as e:
            print("Snapshot failed, will retry next round:", e)
        time.sleep(INTERVAL_SECONDS)