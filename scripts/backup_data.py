import os
import shutil
from datetime import datetime

SOURCE = os.path.join("data", "delay_history.csv")
BACKUP_DIR = os.path.join(os.path.expanduser("~"), "Documents", "eta-backups")

os.makedirs(BACKUP_DIR, exist_ok=True)
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
target = os.path.join(BACKUP_DIR, f"delay_history_{stamp}.csv")

shutil.copy2(SOURCE, target)
print("Saved", target)