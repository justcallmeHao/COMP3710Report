from graphs.plot import plot
from train.data3d import ROOT, read_json
from train.evaluate3d import evaluate
from train.predict3dnormal import predict
from train.train3dnormal import train


def main():
    config = read_json(ROOT / "configs/training3dnormal.json")
    config["labels"] = read_json(ROOT / config["data"]["labels_file"])
    stages = config["pipeline"]
    if stages["train"]:
        train(config)
    if stages["predict"]:
        predict(config)
    if stages["evaluate"]:
        evaluate(config)
    if stages["plot"]:
        plot(config)


if __name__ == "__main__":
    main()
