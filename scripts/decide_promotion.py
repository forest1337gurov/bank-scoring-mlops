import json
from pathlib import Path

import pandas as pd


COMPARISON_PATH = Path("data/model_comparison.csv")
PRODUCTION_METRICS_PATH = Path("ml/artifacts/production_metrics.json")


def main() -> None:
    comparison = pd.read_csv(COMPARISON_PATH)
    candidate = comparison.sort_values(["auc", "gini", "ks"], ascending=False).iloc[0].to_dict()

    production = None
    if PRODUCTION_METRICS_PATH.exists():
        production = json.loads(PRODUCTION_METRICS_PATH.read_text(encoding="utf-8"))

    should_promote = production is None or (
        candidate["auc"] > production["auc"]
        and candidate["gini"] > production["gini"]
        and candidate["ks"] > production["ks"]
    )

    decision = {
        "should_promote": should_promote,
        "candidate": candidate,
        "production": production,
    }

    Path("data").mkdir(exist_ok=True)
    Path("data/promotion_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(decision)


if __name__ == "__main__":
    main()