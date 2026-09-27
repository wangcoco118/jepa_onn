import sys
from pathlib import Path
import yaml
from evals.intphys_test import eval as test_eval

def main():
    config_path = Path(sys.argv[1])
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    test_eval.main(config)

if __name__ == "__main__":
    main()
