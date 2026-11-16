"""Deterministic, clearly fictional demo data for the business query tools."""

import hashlib
import random

GLOBAL_SEED = 20261007

ORDER_STATUSES = ("created", "paid", "shipped", "delivered")
LOGISTICS_STATUSES = ("created", "in_transit", "out_for_delivery", "delivered")
CARRIERS = ("演示快递", "示例物流", "模拟速运")
PRODUCT_CATEGORIES = ("居家演示", "数码演示", "户外演示", "文具演示")


def _derive_rng(namespace: str, identifier: str) -> random.Random:
    """Derive a stable pseudo-random generator from the global seed and an ID."""
    digest = hashlib.sha256(f"{namespace}:{identifier}".encode("utf-8")).hexdigest()
    return random.Random(GLOBAL_SEED ^ int(digest[:16], 16))


def build_order(order_id: str) -> dict:
    """Return stable fictional order data for an order ID."""
    rng = _derive_rng("order", order_id)
    return {
        "demo": True,
        "order_id": order_id,
        "status": rng.choice(ORDER_STATUSES),
        "amount": round(rng.uniform(19.9, 899.0), 2),
        "currency": "CNY",
        "item_count": rng.randint(1, 4),
        "channel": "演示商城",
        "created_at": (
            f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28):02d}"
            f"T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z"
        ),
    }


def build_product(product_id: str) -> dict:
    """Return stable fictional product data for a product ID."""
    rng = _derive_rng("product", product_id)
    category = rng.choice(PRODUCT_CATEGORIES)
    return {
        "demo": True,
        "product_id": product_id,
        "name": f"{category}商品 {rng.randint(1000, 9999)} 号",
        "category": category,
        "price": round(rng.uniform(9.9, 1299.0), 2),
        "currency": "CNY",
        "stock": rng.randint(0, 500),
        "rating": round(rng.uniform(3.5, 5.0), 1),
    }


def build_logistics(order_id: str) -> dict:
    """Return stable fictional logistics data for an order ID."""
    rng = _derive_rng("logistics", order_id)
    status = rng.choice(LOGISTICS_STATUSES)
    reached = LOGISTICS_STATUSES[: LOGISTICS_STATUSES.index(status) + 1]
    return {
        "demo": True,
        "order_id": order_id,
        "carrier": rng.choice(CARRIERS),
        "tracking_no": f"DEMO{rng.randint(10**11, 10**12 - 1)}",
        "status": status,
        "updated_at": f"2026-09-{14 + len(reached):02d}T0{rng.randint(1, 9)}:30:00Z",
        "traces": [
            {
                "status": step,
                "at": f"2026-09-{14 + index:02d}T0{index + 1}:30:00Z",
                "note": f"演示物流节点：{step}",
            }
            for index, step in enumerate(reached)
        ],
    }
