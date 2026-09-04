"""
Generates a synthetic e-commerce interaction dataset for RecoFlow.

Produces:
  data/users.csv        user_id, signup_ts, preferred_categories, price_sensitivity
  data/products.csv     item_id, title, description, category, brand, price, created_ts
  data/events.csv       user_id, item_id, event, timestamp, session_id, device

Event types (with implicit-feedback weights used downstream):
  impression(0.1) < click(1) < wishlist(2) < add_to_cart(3) < purchase(5)
  remove_from_cart is a negative signal.

Run:
  python data/generate_synthetic_data.py [--n-users 2000] [--n-items 500]
                                          [--n-events 60000] [--load-db]
"""
import argparse
import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
random.seed(42)

CATEGORIES = [
    "Electronics", "Books", "Clothing", "Home & Kitchen", "Sports",
    "Beauty", "Toys", "Automotive", "Grocery", "Gaming",
]
BRANDS = {
    "Electronics": ["Sennheiser", "Boat", "Sony", "Anker", "JBL"],
    "Books": ["Penguin", "HarperCollins", "Scholastic"],
    "Clothing": ["Levis", "H&M", "Zara", "Uniqlo"],
    "Home & Kitchen": ["Prestige", "Philips", "IKEA"],
    "Sports": ["Nike", "Adidas", "Puma", "Decathlon"],
    "Beauty": ["Nivea", "Lakme", "Mamaearth"],
    "Toys": ["LEGO", "Hasbro", "Funskool"],
    "Automotive": ["Bosch", "Michelin", "3M"],
    "Grocery": ["Tata", "Nestle", "Amul"],
    "Gaming": ["Logitech", "Razer", "SteelSeries", "Sony"],
}
PRODUCT_ADJ = ["Wireless", "Pro", "Ultra", "Lite", "Max", "Classic", "Portable", "Smart"]
PRODUCT_NOUN = {
    "Electronics": ["Headphones", "Earbuds", "Speaker", "Charger", "Cable", "Power Bank"],
    "Books": ["Novel", "Cookbook", "Biography", "Guide"],
    "Clothing": ["T-Shirt", "Jacket", "Jeans", "Sneakers"],
    "Home & Kitchen": ["Mixer", "Cookware Set", "Air Fryer", "Lamp"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Running Shoes", "Water Bottle"],
    "Beauty": ["Face Wash", "Moisturizer", "Lipstick", "Sunscreen"],
    "Toys": ["Building Blocks", "Action Figure", "Puzzle", "Board Game"],
    "Automotive": ["Car Vacuum", "Tire Inflator", "Dash Cam", "Seat Cover"],
    "Grocery": ["Snack Pack", "Tea Box", "Coffee Beans", "Cereal"],
    "Gaming": ["Mouse", "Keyboard", "Headset", "Controller"],
}

EVENT_WEIGHTS = {
    "impression": 0.1, "click": 1.0, "search": 0.2, "category_visit": 0.3,
    "wishlist": 2.0, "add_to_cart": 3.0, "remove_from_cart": -1.5, "purchase": 5.0,
}
# funnel probabilities conditioned on a shown item
FUNNEL = [
    ("impression", 1.0),
    ("click", 0.35),
    ("wishlist", 0.08),
    ("add_to_cart", 0.15),
    ("remove_from_cart", 0.04),
    ("purchase", 0.06),
]


def gen_products(n_items):
    rows = []
    start = datetime(2025, 1, 1)
    for i in range(n_items):
        cat = random.choice(CATEGORIES)
        brand = random.choice(BRANDS[cat])
        title = f"{random.choice(PRODUCT_ADJ)} {random.choice(PRODUCT_NOUN[cat])}"
        price = round(float(RNG.lognormal(mean=6.5, sigma=0.9)), 2)
        created = start + timedelta(days=int(RNG.integers(0, 600)))
        rows.append({
            "item_id": i,
            "title": title,
            "description": f"{title} by {brand}. Category: {cat}. A great choice for {cat.lower()} needs.",
            "category": cat,
            "brand": brand,
            "price": price,
            "created_ts": created.isoformat(),
        })
    return pd.DataFrame(rows)


def gen_users(n_users):
    rows = []
    start = datetime(2025, 1, 1)
    for u in range(n_users):
        prefs = random.sample(CATEGORIES, k=random.randint(1, 3))
        rows.append({
            "user_id": u,
            "signup_ts": (start + timedelta(days=int(RNG.integers(0, 500)))).isoformat(),
            "preferred_categories": json.dumps(prefs),
            "price_sensitivity": round(float(RNG.uniform(0.2, 1.0)), 2),
        })
    return pd.DataFrame(rows)


def gen_events(users_df, products_df, n_events):
    rows = []
    devices = ["mobile", "desktop", "tablet"]
    start = datetime(2025, 6, 1)
    horizon_days = 95  # ~3 months, so we can do a clean temporal split
    prod_by_cat = products_df.groupby("category")["item_id"].apply(list).to_dict()

    session_counter = 0
    events_made = 0
    while events_made < n_events:
        u = users_df.sample(1, random_state=int(RNG.integers(0, 1_000_000))).iloc[0]
        prefs = json.loads(u["preferred_categories"])
        cat = random.choice(prefs) if random.random() < 0.75 else random.choice(CATEGORIES)
        candidates = prod_by_cat.get(cat, products_df["item_id"].tolist())
        session_counter += 1
        session_id = f"s{session_counter}"
        day_offset = int(RNG.integers(0, horizon_days))
        session_start = start + timedelta(days=day_offset, hours=int(RNG.integers(0, 24)))
        device = random.choice(devices)

        n_shown = random.randint(2, 8)
        shown_items = random.sample(candidates, k=min(n_shown, len(candidates)))

        for offset, item_id in enumerate(shown_items):
            ts = session_start + timedelta(minutes=offset * 2)
            rows.append({
                "user_id": int(u["user_id"]), "item_id": int(item_id), "event": "impression",
                "timestamp": ts.isoformat(), "session_id": session_id, "device": device,
            })
            events_made += 1
            for ev, p in FUNNEL[1:]:
                if random.random() < p:
                    ts2 = ts + timedelta(seconds=random.randint(5, 300))
                    rows.append({
                        "user_id": int(u["user_id"]), "item_id": int(item_id), "event": ev,
                        "timestamp": ts2.isoformat(), "session_id": session_id, "device": device,
                    })
                    events_made += 1
            if events_made >= n_events:
                break
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=2000)
    ap.add_argument("--n-items", type=int, default=500)
    ap.add_argument("--n-events", type=int, default=60000)
    ap.add_argument("--out-dir", type=str, default=os.path.dirname(__file__))
    ap.add_argument("--load-db", action="store_true", help="also load into PostgreSQL via src/db")
    args = ap.parse_args()

    users = gen_users(args.n_users)
    products = gen_products(args.n_items)
    events = gen_events(users, products, args.n_events)

    users.to_csv(os.path.join(args.out_dir, "users.csv"), index=False)
    products.to_csv(os.path.join(args.out_dir, "products.csv"), index=False)
    events.to_csv(os.path.join(args.out_dir, "events.csv"), index=False)

    print(f"users={len(users)} products={len(products)} events={len(events)}")
    print(events["event"].value_counts())

    if args.load_db:
        from src.db.load import load_all
        load_all(users, products, events)


if __name__ == "__main__":
    main()
