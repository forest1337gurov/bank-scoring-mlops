from pathlib import Path

SQL_DIR = Path(__file__).parent.parent / "sql"


def load_sql(filename: str) -> str:
    """
    Загружает SQL-запрос из файла.
    """
    with open(SQL_DIR / filename, "r", encoding="utf-8") as file:
        return file.read()