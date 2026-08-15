import json
import shutil
from pathlib import Path

import pandas as pd


COMPARISON_PATH = Path("data/model_comparison.csv")
PRODUCTION_METRICS_PATH = Path("ml/artifacts/production_metrics.json")


def main():
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

    if not should_promote:
        print("No promotion. Candidate is not better than production.")
        return

    src_model = Path(candidate["model_path"])
    src_bundle = src_model.with_name("catboost_bundle.pkl")

    dst_model = Path("ml/artifacts/production_model.cbm")
    dst_bundle = Path("ml/artifacts/production_bundle.pkl")

    shutil.copy2(src_model, dst_model)
    shutil.copy2(src_bundle, dst_bundle)

    production_payload = {
        **candidate,
        "model_path": str(dst_model),
        "bundle_path": str(dst_bundle),
    }

    PRODUCTION_METRICS_PATH.write_text(
        json.dumps(production_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Promoted:", production_payload)


if __name__ == "__main__":
    main()