from config.clickhouse import get_client


def main():
    client = get_client()

    result = client.query("SELECT 1")

    print(result.result_rows)


if __name__ == "__main__":
    main()