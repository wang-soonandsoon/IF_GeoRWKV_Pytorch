from __future__ import annotations

from typing import List, Optional


# Paper palette protocol (label->RGB) used by `prepare_paper_figure_tree.py`.
# Notes:
# - Labels are 1-based; 0 means unlabeled/background.
# - Each entry is RGB (0..255) for label_id-1.

HOUSTON2013_COLOR_MAP: List[List[int]] = [
    [0, 0, 131],
    [0, 0, 203],
    [0, 19, 255],
    [0, 91, 255],
    [0, 167, 255],
    [0, 239, 255],
    [55, 255, 199],
    [131, 255, 123],
    [203, 255, 51],
    [255, 235, 0],
    [255, 163, 0],
    [255, 87, 0],
    [255, 15, 0],
    [199, 0, 0],
    [127, 0, 0],
]

TRENTO_COLOR_MAP: List[List[int]] = [
    [0, 47, 255],
    [0, 223, 255],
    [143, 255, 111],
    [255, 207, 0],
    [255, 31, 0],
    [127, 0, 0],
]

HOUSTON2013_CLASS_NAMES: List[str] = [
    "Healthy grass",
    "Stressed grass",
    "Synthetic grass",
    "Trees",
    "Soil",
    "Water",
    "Residential",
    "Commercial",
    "Road",
    "Highway",
    "Railway",
    "Parking lot 1",
    "Parking lot 2",
    "Tennis court",
    "Running track",
]

TRENTO_CLASS_NAMES: List[str] = [
    "Apple trees",
    "Buildings",
    "Ground",
    "Woods",
    "Vineyard",
    "Roads",
]


_PALETTE_BY_DATASET = {
    "Houston2013": {"color_map": HOUSTON2013_COLOR_MAP, "class_names": HOUSTON2013_CLASS_NAMES},
    "Trento": {"color_map": TRENTO_COLOR_MAP, "class_names": TRENTO_CLASS_NAMES},
}


def get_label_color_map(dataset: str) -> Optional[List[List[int]]]:
    entry = _PALETTE_BY_DATASET.get(str(dataset))
    if not isinstance(entry, dict):
        return None
    cmap = entry.get("color_map")
    return cmap if isinstance(cmap, list) else None


def get_class_names(dataset: str) -> Optional[List[str]]:
    entry = _PALETTE_BY_DATASET.get(str(dataset))
    if not isinstance(entry, dict):
        return None
    names = entry.get("class_names")
    return names if isinstance(names, list) else None

