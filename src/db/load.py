"""
Load the synthetic dataset into PostgreSQL. Kept separate from
generate_synthetic_data.py so the ML core never has a hard dependency on a
running database.

Env vars (or .env): POSTGRES_URL, default
  postgresql://postgres:postgres@localhost:5432/recoflow
"""
import os

from sqlalchemy import create_engine, text

POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/recoflow"
)


def get_engine():
    return create_engine(POSTGRES_URL)


def apply_schema(engine):
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path) as f:
        ddl = f.read()
    with engine.begin() as conn:
        for stmt in ddl.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))


def load_all(users_df, products_df, events_df):
    engine = get_engine()
    apply_schema(engine)
    users_df.to_sql("users", engine, if_exists="append", index=False,
                     method="multi", chunksize=1000)
    products_df.to_sql("products", engine, if_exists="append", index=False,
                        method="multi", chunksize=1000)
    events_df.to_sql("events", engine, if_exists="append", index=False,
                      method="multi", chunksize=2000)
    print("Loaded users/products/events into PostgreSQL.")


if __name__ == "__main__":
    import pandas as pd
    base = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    load_all(
        pd.read_csv(os.path.join(base, "users.csv")),
        pd.read_csv(os.path.join(base, "products.csv")),
        pd.read_csv(os.path.join(base, "events.csv")),
    )
