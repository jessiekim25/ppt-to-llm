from contextlib import contextmanager

import pymysql

DDL = """
CREATE TABLE IF NOT EXISTS brand_guidelines (
  id          INT AUTO_INCREMENT PRIMARY KEY,
  product     VARCHAR(255),
  codename    VARCHAR(255),
  section     VARCHAR(255),
  detail      TEXT,
  model       VARCHAR(255),
  sub_section VARCHAR(255),
  image_path  VARCHAR(1024)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
"""

INSERT = """
INSERT INTO brand_guidelines
  (product, codename, section, detail, model, sub_section, image_path)
VALUES
  (%(product)s, %(codename)s, %(section)s, %(detail)s, %(model)s, %(sub_section)s, %(image_path)s);
"""


@contextmanager
def connect(host: str, port: int, user: str, password: str, database: str):
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
    )
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)


def insert_row(conn, row: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(INSERT, row)
