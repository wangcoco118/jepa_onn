import sys
import yaml
from evals.intphys_test import eval as test_eval

if __name__ == "__main__":
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("data", {})["batch_size"] = int(sys.argv[2])
    config["output_dir"] = sys.argv[3]
    test_eval.main(config)
