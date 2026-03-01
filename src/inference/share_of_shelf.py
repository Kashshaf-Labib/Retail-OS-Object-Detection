"""Share of Shelf analytics — computes SKU distribution from detections."""

from collections import Counter

from src.config import CLASS_NAMES


def compute_share_of_shelf(detections: list[dict]) -> dict:
    """
    Compute the percentage share of shelf for each detected SKU.

    Args:
        detections: List of detection dicts with 'class_name' and 'confidence'

    Returns:
        dict with 'total_products', 'sku_counts', and 'sku_percentages'
    """
    if not detections:
        return {
            "total_products": 0,
            "sku_counts": {},
            "sku_percentages": {},
            "top_skus": [],
        }

    # Count occurrences of each SKU
    class_names = [d["class_name"] for d in detections]
    counts = Counter(class_names)
    total = sum(counts.values())

    # Compute percentages and sort by count
    sku_data = []
    for sku_name, count in counts.most_common():
        percentage = round((count / total) * 100, 2)
        sku_data.append({
            "sku": sku_name,
            "count": count,
            "percentage": percentage,
        })

    # Build response
    sku_counts = {item["sku"]: item["count"] for item in sku_data}
    sku_percentages = {item["sku"]: item["percentage"] for item in sku_data}

    return {
        "total_products": total,
        "sku_counts": sku_counts,
        "sku_percentages": sku_percentages,
        "top_skus": sku_data[:10],  # Top 10 for quick view
        "all_skus": sku_data,
    }
