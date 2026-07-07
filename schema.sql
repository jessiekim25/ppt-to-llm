CREATE DATABASE IF NOT EXISTS jihwi
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS jihwi.brand_guidelines (
  id          INT AUTO_INCREMENT PRIMARY KEY,
  page        INT,
  product     VARCHAR(255),
  codename    VARCHAR(255),
  section     VARCHAR(255),
  detail      TEXT,
  model       VARCHAR(255),
  sub_section VARCHAR(255),
  image_path  VARCHAR(1024)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
