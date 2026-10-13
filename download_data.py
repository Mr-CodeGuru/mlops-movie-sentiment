from pathlib import Path
import shutil
import sys

import kagglehub

SLUG = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
TARGET = Path(__file__).parent / "data" / "imdb.csv"


def main() -> None:
    TARGET.parent.mkdir(exist_ok=True)
    src_dir = Path(kagglehub.dataset_download(SLUG))
    csvs = list(src_dir.rglob("*.csv"))
    if not csvs:
        sys.exit(f"No CSV found in {src_dir}")
    shutil.copy(csvs[0], TARGET)
    print(f"Dataset saved to {TARGET}")


if __name__ == "__main__":
    main()