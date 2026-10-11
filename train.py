from chap_model import train_model

import sys
from pathlib import Path


if __name__ == "__main__":
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    train_model(sys.argv[1], sys.argv[2])
