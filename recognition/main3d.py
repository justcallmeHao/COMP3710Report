from graphs.plot import plot
from train.data3d import ROOT, read_json
from train.evaluate3d import evaluate
from train.predict3d import predict
from train.pipeline import run_pipeline
from train.train3d import train


def main():
    config = read_json(ROOT / "configs/training3d.json")
    config["labels"] = read_json(ROOT / config["data"]["labels_file"])
    run_pipeline(config, train=train, predict=predict, evaluate=evaluate, plot=plot)


if __name__ == "__main__":
    main()
