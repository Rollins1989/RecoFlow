"""
Pure update function: one event in, mutated profile out. Kept separate from
the consumer loop so it's trivially unit-testable and reusable from a Flink
Python UDF later.
"""
from __future__ import annotations

EVENT_WEIGHTS = {
    "impression": 0.1, "click": 1.0, "search": 0.2, "category_visit": 0.3,
    "wishlist": 2.0, "add_to_cart": 3.0, "remove_from_cart": -1.5, "purchase": 5.0,
}
RECENT_INTERESTS_MAXLEN = 10


def apply_event_to_profile(profile_store: dict, event: dict) -> dict:
    user_id = event["user_id"]
    profile = profile_store.setdefault(user_id, {
        "category_affinity": {}, "recent_interests": [], "last_event_ts": None,
    })

    weight = EVENT_WEIGHTS.get(event.get("event"), 0.0)
    category = event.get("category")  # optional, if the caller enriches the event
    if category:
        profile["category_affinity"][category] = (
            profile["category_affinity"].get(category, 0.0) + weight
        )

    interest = event.get("query") or category or event.get("item_id")
    if interest is not None:
        profile["recent_interests"].append(interest)
        profile["recent_interests"] = profile["recent_interests"][-RECENT_INTERESTS_MAXLEN:]

    profile["last_event_ts"] = event.get("timestamp")
    return profile
