from typing import Optional, Set

_POSITVE_TOKENS: Set[str] = {
    "great",
    "good",
    "like",
    "love",
    "just",
    "use",
    "product",
    "easy",
    "really",
    "little",
}

_NEGATIVE_TOKENS: Set[str] = {
    "product",
    "like",
    "just",
    "work",
    "use",
    "don",
    "did",
    "time",
    "good",
    "didn",
}


def classify(
    review: str,
    positive_tokens: Optional[Set[str]] = None,
    negative_tokens: Optional[Set[str]] = None,
) -> int:
    """
    Classify a review as positive (1) or negative (0).
    """
    if positive_tokens is None:
        positive_tokens = _POSITVE_TOKENS
    if negative_tokens is None:
        negative_tokens = _NEGATIVE_TOKENS

    review_tokens = set(review.lower().split())
    positive_tokens_in_review = len(review_tokens.intersection(positive_tokens))
    negative_tokens_in_review = len(review_tokens.intersection(negative_tokens))

    if positive_tokens_in_review > negative_tokens_in_review:
        return 1
    return 0
