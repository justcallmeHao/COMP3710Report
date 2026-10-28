import json

from train.data2d import ROOT
from train.evaluate2d import evaluate
from train.predict2d import predict
from train.train2d import train


def main():
    config = json.loads((ROOT / "configs/training2d.json").read_text(encoding="utf-8"))
    stages = config["pipeline"]
    if stages["train"]:
        train(config)
    if stages["predict"]:
        predict(config)
    if stages["evaluate"]:
        evaluate(config)


if __name__ == "__main__":
    main()
