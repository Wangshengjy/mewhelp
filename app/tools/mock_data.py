"""Deterministic, clearly fictional demo data for the business query tools."""

import hashlib
import random

GLOBAL_SEED = 20261007

ORDER_STATUSES = ("created", "paid", "shipped", "delivered")
LOGISTICS_STATUSES = ("created", "in_transit", "out_for_delivery", "delivered")
CARRIERS = ("演示快递", "示例物流", "模拟速运")
PRODUCT_CATEGORIES = ("居家演示", "数码演示", "户外演示", "文具演示")
PRODUCT_NAMES = ("智能猫砂盆", "自动饮水机", "猫爬架", "主粮五连包")


def _derive_rng(namespace: str, identifier: str) -> random.Random:
    """Derive a stable pseudo-random generator from the global seed and an ID."""
    digest = hashlib.sha256(f"{namespace}:{identifier}".encode("utf-8")).hexdigest()
    return random.Random(GLOBAL_SEED ^ int(digest[:16], 16))


def build_order(order_id: str) -> dict:
    """Return stable fictional order data for an order ID."""
    rng = _derive_rng("order", order_id)
    # ch05 D7 · tracking_no 只能从 query_order 的返回里取到，query_logistics 吃它——
    # 造出「先查订单再查物流」的真实数据依赖，让 ReAct 多步成立。
    tracking_rng = _derive_rng("order-tracking", order_id)
    product_rng = _derive_rng("order-product", order_id)
    return {
        "demo": True,
        "order_id": order_id,
        "product": f"{product_rng.choice(PRODUCT_NAMES)}×{product_rng.randint(1, 2)}",
        "status": rng.choice(ORDER_STATUSES),
        "amount": round(rng.uniform(19.9, 899.0), 2),
        "currency": "CNY",
        "item_count": rng.randint(1, 4),
        "channel": "演示商城",
        "tracking_no": f"DEMO{tracking_rng.randint(10**11, 10**12 - 1)}",
        "created_at": (
            f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28):02d}"
            f"T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z"
        ),
    }


def list_user_orders(user_id: str) -> list[dict]:
    """ch06 · 用户订单列表(interrupt 选单卡片用):每单与 query_order 同源。

    按派生 rng 生成 2~4 个 4 位演示单号,快照直接复用 build_order——
    前端选中回填 order_id 后,fetch_order 再查的必然是同一份确定性数据。
    """
    rng = _derive_rng("user_orders", user_id or "")
    count = rng.randint(2, 4)
    ids: list[str] = []
    while len(ids) < count:
        candidate = str(rng.randint(1000, 9999))
        if candidate not in ids:
            ids.append(candidate)
    orders = [build_order(order_id) for order_id in ids]
    return [
        {
            "order_id": order["order_id"],
            "product": order["product"],
            "status": order["status"],
            "amount": order["amount"],
        }
        for order in orders
    ]


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


def build_logistics(tracking_no: str) -> dict:
    """Return stable fictional logistics data for a tracking number (ch05 D7)."""
    rng = _derive_rng("logistics", tracking_no)
    status = rng.choice(LOGISTICS_STATUSES)
    reached = LOGISTICS_STATUSES[: LOGISTICS_STATUSES.index(status) + 1]
    return {
        "demo": True,
        "tracking_no": tracking_no,
        "carrier": rng.choice(CARRIERS),
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
