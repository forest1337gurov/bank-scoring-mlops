import json
import shutil
from pathlib import Path


DECISION_PATH = Path("data/promotion_decision.json")
PRODUCTION_METRICS_PATH = Path("ml/artifacts/production_metrics.json")


def main() -> None:
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))

    if not decision["should_promote"]:
        print("No promotion.")
        return

    candidate = decision["candidate"]

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