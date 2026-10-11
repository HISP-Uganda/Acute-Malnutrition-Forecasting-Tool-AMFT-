from chap_model import predict_model

import sys
from pathlib import Path


if __name__ == "__main__":
    Path(sys.argv[4]).parent.mkdir(parents=True, exist_ok=True)
    predict_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
