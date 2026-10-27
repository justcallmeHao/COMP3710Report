from copy import deepcopy

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
    for fold in config["inference"]["folds"]:
        request = deepcopy(config)
        request["inference"]["fold"] = fold
        request["inference"]["prediction_id"] += f"_fold{fold}"
        request["graphs"]["figure_id"] += f"_fold{fold}"
        request["graphs"]["folds"] = [fold] if fold in config["graphs"]["folds"] else []
        predictions = predict(request, evaluate_results=stages["evaluate"]) if stages["predict"] else None
        if stages["evaluate"]:
            evaluate(request, predictions)
        if stages["plot"]:
            plot(request)


if __name__ == "__main__":
    main()
