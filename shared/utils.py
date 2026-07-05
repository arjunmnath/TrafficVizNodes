import logging
import numpy as np


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.shape != vec2.shape:
        return 0.0
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def compute_attribute_similarity(attr1: dict, attr2: dict) -> float:
    score = 0.0
    total_attrs = 0
    if "color" in attr1 and "color" in attr2:
        total_attrs += 1
        if attr1["color"] == attr2["color"]:
            score += 1.0
    if (
        "type" in attr1
        and attr1["type"] is not None
        and "type" in attr2
        and attr2["type"] is not None
    ):
        total_attrs += 1
        if attr1["type"] == attr2["type"]:
            score += 1.0

    if total_attrs == 0:
        return 0.5
    return score / total_attrs
