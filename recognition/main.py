from graphs.plot import plot
from train.data import ROOT, read_json
from train.evaluate import evaluate
from train.predict import predict
from train.train import train


def main():
    config = read_json(ROOT / "configs/training.json")
    config["labels"] = read_json(ROOT / config["data"]["labels_file"])
    stages = config["pipeline"]
    if stages["train"]:
        train(config)
    predictions = predict(config, evaluate_results=stages["evaluate"]) if stages["predict"] else None
    if stages["evaluate"]:
        evaluate(config, predictions)
    if stages["plot"]:
        plot(config)


if __name__ == "__main__":
    main()
