from graphs.plot import plot
from train.data import ROOT, discover, read_json, split_manifest
from train.evaluate import evaluate
from train.predict import predict
from train.train import train


def main():
    config = read_json(ROOT / "configs/training.json")
    config["labels"] = read_json(ROOT / config["data"]["labels_file"])
    stages = config["pipeline"]
    prepared = None
    if stages["prepare"]:
        cases = discover(config)
        splits = split_manifest(config, cases)
        prepared = cases, splits
        print(f"Prepared splits for {len(cases)} scans, {len(splits['definition']['patients'])} patients.")
    if stages["train"]:
        train(config, prepared)
    predictions = predict(config, evaluate_results=stages["evaluate"]) if stages["predict"] else None
    if stages["evaluate"]:
        evaluate(config, predictions)
    if stages["plot"]:
        plot(config)


if __name__ == "__main__":
    main()
