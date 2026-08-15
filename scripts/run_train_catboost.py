from ml.train_catboost import CatBoostTrainer


def main():
    trainer = CatBoostTrainer()
    trainer.train()


if __name__ == "__main__":
    main()