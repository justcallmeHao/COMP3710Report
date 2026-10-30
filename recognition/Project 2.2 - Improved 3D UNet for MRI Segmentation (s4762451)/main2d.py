import json

from graphs.plot import plot
from helpers.data2d import ROOT
from helpers.evaluate2d import evaluate
from helpers.predict2d import predict
from helpers.pipeline import run_pipeline
from helpers.train2d import train


def main():
    config = json.loads((ROOT / "configs/training2d.json").read_text(encoding="utf-8"))
    run_pipeline(config, train=train, predict=predict, evaluate=evaluate, plot=plot)


if __name__ == "__main__":
    main()
