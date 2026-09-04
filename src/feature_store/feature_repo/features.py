"""
Feast feature definitions. The point of a feature store here is
training-serving consistency: `src/features/feature_engineering.py` computes
these exact columns offline (for training data) and Feast serves the same
columns online (for the API's `<50ms` p99 lookup) — so the model never sees
a distribution shift caused by the batch/online pipelines drifting apart.

    Raw events -> feature computation -> Feature Store -> {offline, online}
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

user = Entity(name="user_id", value_type=ValueType.INT64, description="e-commerce user")
item = Entity(name="item_id", value_type=ValueType.INT64, description="product")

user_features_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
)
item_features_source = FileSource(
    path="data/item_features.parquet",
    timestamp_field="event_timestamp",
)

user_feature_view = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="n_views", dtype=Int64),
        Field(name="n_purchases", dtype=Int64),
        Field(name="avg_price_seen", dtype=Float32),
        Field(name="is_cold_start", dtype=Int64),
    ],
    online=True,
    source=user_features_source,
)

item_feature_view = FeatureView(
    name="item_features",
    entities=[item],
    ttl=timedelta(days=7),
    schema=[
        Field(name="ctr", dtype=Float32),
        Field(name="purchase_rate", dtype=Float32),
        Field(name="freshness", dtype=Float32),
        Field(name="category", dtype=String),
        Field(name="price", dtype=Float32),
    ],
    online=True,
    source=item_features_source,
)
