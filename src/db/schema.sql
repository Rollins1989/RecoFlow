-- RecoFlow PostgreSQL schema

CREATE TABLE IF NOT EXISTS users (
    user_id             BIGINT PRIMARY KEY,
    signup_ts           TIMESTAMP NOT NULL,
    preferred_categories JSONB,
    price_sensitivity   REAL
);

CREATE TABLE IF NOT EXISTS products (
    item_id     BIGINT PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT,
    category    TEXT NOT NULL,
    brand       TEXT,
    price       NUMERIC(10, 2),
    created_ts  TIMESTAMP NOT NULL,
    is_active   BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS events (
    event_id    BIGSERIAL PRIMARY KEY,
    user_id     BIGINT REFERENCES users(user_id),
    item_id     BIGINT REFERENCES products(item_id),
    event       TEXT NOT NULL CHECK (event IN
        ('impression','click','view','add_to_cart','wishlist',
         'purchase','remove_from_cart','search','category_visit')),
    session_id  TEXT,
    device      TEXT,
    timestamp   TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_item ON events(item_id);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id             BIGINT PRIMARY KEY REFERENCES users(user_id),
    category_affinity   JSONB,      -- {"Electronics": 0.92, "Books": 0.34, ...}
    price_pref_low      NUMERIC(10,2),
    price_pref_high     NUMERIC(10,2),
    preferred_brands    JSONB,
    recent_interests    JSONB,      -- rolling list of recent categories/keywords
    updated_at          TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recommendations (
    request_id      BIGSERIAL PRIMARY KEY,
    user_id         BIGINT,
    item_id         BIGINT,
    rank            INT,
    score           REAL,
    reason          TEXT,
    candidate_source TEXT,
    model_version   TEXT,
    generated_at    TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    variant         TEXT NOT NULL,     -- 'control' | 'treatment'
    model_version   TEXT,
    started_at      TIMESTAMP DEFAULT now(),
    ended_at        TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version   TEXT PRIMARY KEY,
    stage           TEXT NOT NULL,     -- 'staging' | 'production' | 'archived'
    ndcg_at_10      REAL,
    recall_at_10    REAL,
    trained_at      TIMESTAMP DEFAULT now(),
    mlflow_run_id   TEXT
);
