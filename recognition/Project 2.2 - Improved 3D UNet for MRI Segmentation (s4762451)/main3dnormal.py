from graphs.plot import plot
from dataset import ROOT, read_json
from helpers.evaluate3d import evaluate
from helpers.predict3dnormal import predict
from helpers.pipeline import run_pipeline
from helpers.train3dnormal import train


def main():
    """load the baseline 3D config and labels and run the enabled pipeline stages"""
    config = read_json(ROOT / "configs/training3dnormal.json")
    config["labels"] = read_json(ROOT / config["data"]["labels_file"])
    run_pipeline(config, train=train, predict=predict, evaluate=evaluate, plot=plot)


if __name__ == "__main__":
    main()
