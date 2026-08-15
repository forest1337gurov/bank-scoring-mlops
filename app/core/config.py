from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Bank Scoring API"
    PROJECT_VERSION: str = "0.1.0"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    CLICKHOUSE_HOST: str
    CLICKHOUSE_PORT: int = 8123

    CLICKHOUSE_USER: str
    CLICKHOUSE_PASSWORD: str

    CLICKHOUSE_DATABASE: str

    TRAIN_START_DATE: str = "2025-06-01"
    TRAIN_END_DATE: str = "2025-08-01"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()