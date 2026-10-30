import json

from helpers.preprocess2D import preprocess as preprocess2d
from helpers.preprocess3D import ROOT, preprocess as preprocess3d


def main():
    """prepare cached slices and volumes for all three models"""
    config = json.loads((ROOT / "configs/training3d.json").read_text(encoding="utf-8"))
    config["labels"] = json.loads(
        (ROOT / config["data"]["labels_file"]).read_text(encoding="utf-8")
    )
    print("Load files for 2d model")
    preprocess2d()
    # share for both 3d improved and 3d normal
    print("Load files for 3d models")
    preprocess3d(config)


if __name__ == "__main__":
    main()
