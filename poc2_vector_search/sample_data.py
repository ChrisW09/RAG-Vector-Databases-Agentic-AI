"""Generate a synthetic product catalog and persist it to data/products.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    "Apparel": [
        "sweater", "t-shirt", "jacket", "hoodie", "trousers", "dress", "scarf",
    ],
    "Electronics": [
        "wireless headphones", "smartwatch", "bluetooth speaker", "laptop",
        "tablet", "mechanical keyboard", "monitor",
    ],
    "Home": [
        "scented candle", "ceramic mug", "throw blanket", "table lamp",
        "wall clock", "cushion", "wooden shelf",
    ],
    "Sports": [
        "yoga mat", "running shoes", "dumbbell set", "tennis racket",
        "cycling helmet", "fitness tracker",
    ],
    "Books": [
        "novel", "cookbook", "biography", "history book", "science textbook",
        "poetry collection",
    ],
}

ADJECTIVES = [
    "cosy", "lightweight", "ergonomic", "premium", "compact", "minimalist",
    "rugged", "elegant", "vintage", "modern", "eco-friendly", "handcrafted",
    "wireless", "portable", "noise-cancelling", "breathable", "soft",
]

USE_CASES = [
    "perfect for the office", "ideal for travelling", "great for outdoor use",
    "designed for everyday wear", "suitable for gifts", "made for athletes",
    "built for long battery life", "engineered for studio quality sound",
    "loved by readers", "tested for durability",
]


def get_products(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        category = rng.choice(list(CATEGORIES.keys()))
        base = rng.choice(CATEGORIES[category])
        adj1 = rng.choice(ADJECTIVES)
        adj2 = rng.choice(ADJECTIVES)
        use = rng.choice(USE_CASES)
        title = f"{adj1.capitalize()} {base}".strip()
        description = (
            f"A {adj1} and {adj2} {base} {use}. "
            f"This {category.lower()} item offers great value and quality."
        )
        price = float(np.round(rng.uniform(9.99, 499.99), 2))
        rows.append(
            {
                "id": f"P{i:05d}",
                "title": title,
                "description": description,
                "category": category,
                "price": price,
            }
        )
    df = pd.DataFrame(rows)
    out = Path("data/products.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    df = get_products()
    print(f"Wrote {len(df)} products to data/products.csv")
