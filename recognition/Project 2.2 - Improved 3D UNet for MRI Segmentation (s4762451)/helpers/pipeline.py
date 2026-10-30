from copy import deepcopy


def run_pipeline(config, *, train, predict, evaluate, plot):
    """run enabled stages with separate validation and test outputs"""
    stages = config["pipeline"]
    if stages["train"]:
        train(config)
    if stages["predict"]:
        predict(config)
    if stages["evaluate"]:
        evaluate(config)
    if stages.get("test", False):
        test_config = deepcopy(config)
        settings = config.get("testing", {})
        test_config["inference"].update(
            partition="test", checkpoint="best",
            prediction_id=settings.get("prediction_id", "test_v1"))
        test_config["graphs"]["figure_id"] = settings.get("figure_id", "test_figures_v1")
        predict(test_config)
        evaluate(test_config)
    if stages["plot"]:
        plot(config)
        if stages.get("test", False):
            plot(test_config, include_training=False)
