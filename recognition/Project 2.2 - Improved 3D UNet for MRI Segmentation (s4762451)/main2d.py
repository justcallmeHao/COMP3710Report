import json

from graphs.plot import plot
from train.data2d import ROOT
from train.evaluate2d import evaluate
from train.predict2d import predict
from train.pipeline import run_pipeline
from train.train2d import train


def main():
    config = json.loads((ROOT / "configs/training2d.json").read_text(encoding="utf-8"))
    run_pipeline(config, train=train, predict=predict, evaluate=evaluate, plot=plot)


if __name__ == "__main__":
    main()
