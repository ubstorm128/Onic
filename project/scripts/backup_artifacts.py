"""
Backup reports and metrics before applying corrections.
"""
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

backup_dir = PROJECT_ROOT / "artifacts" / "backups"
reports_backup = backup_dir / "reports_backup"
metrics_backup = backup_dir / "metrics_backup"

reports_backup.mkdir(parents=True, exist_ok=True)
metrics_backup.mkdir(parents=True, exist_ok=True)

reports_src = PROJECT_ROOT / "reports"
metrics_src = PROJECT_ROOT / "artifacts" / "metrics"

for f in reports_src.iterdir():
    if f.is_file():
        shutil.copy2(f, reports_backup / f.name)

for f in metrics_src.iterdir():
    if f.is_file():
        shutil.copy2(f, metrics_backup / f.name)

print(f"Backed up {len(list(reports_backup.iterdir()))} reports to {reports_backup}")
print(f"Backed up {len(list(metrics_backup.iterdir()))} metrics to {metrics_backup}")
