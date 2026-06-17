import kagglehub
from pathlib import Path
import shutil

# downloads the pima dataset to local directory

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

download_path = kagglehub.dataset_download(
    "uciml/pima-indians-diabetes-database",
    output_dir=DATA_DIR
)

download_path = Path(download_path)

for csv_file in download_path.glob("*.csv"):
    shutil.move(csv_file, DATA_DIR / "pima.csv")
    break

complete_dir = DATA_DIR / ".complete"
if complete_dir.exists():
    shutil.rmtree(complete_dir)

pima_path = DATA_DIR / "pima.csv"

with open(pima_path, "r") as f:
    lines = f.readlines()

with open(pima_path, "w") as f:
    f.writelines(lines[1:])

print("Dataset saved to:", DATA_DIR / "pima.csv")