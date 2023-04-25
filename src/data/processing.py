from typing import Optional


def star_to_label(star: int) -> Optional[int]:
    """
    Convert star rating to label. If star is 3, return None.
    If star is 4 or 5, return 1. Otherwise, return 0.
    """
    if star == 3:
        return None
    if star >= 4:
        return 1
    else:
        return 0
