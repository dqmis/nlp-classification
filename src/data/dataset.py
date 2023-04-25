from typing import List, Optional, Tuple, Union

import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.processing import star_to_label


def _process_dataset_to_data_frame(
    dataset: datasets.Dataset, use_stars: bool = False
) -> pd.DataFrame:
    dataset_df = pd.DataFrame(dataset).drop(
        [
            "review_id",
            "product_id",
            "reviewer_id",
            "review_title",
            "product_category",
        ],
        axis=1,
    )

    if not use_stars:
        dataset_df["label"] = dataset_df["stars"].apply(star_to_label)
        dataset_df.dropna(inplace=True)
        dataset_df["label"] = dataset_df["label"].astype(int)
        dataset_df.drop("stars", axis=1, inplace=True)

    return dataset_df


def _process_articles_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.text = df.text.apply(lambda x: x.lower().replace("reuters", ""))
    return df


def load_articles_dataset(
    n_sample: Optional[int] = None, process: bool = False
) -> pd.DataFrame:
    dataset = datasets.load_dataset("GonzaloA/fake_news")
    df = pd.DataFrame(dataset["train"])
    df.drop(columns=["Unnamed: 0", "title"], inplace=True)
    if process:
        df = _process_articles_dataset(df)
    if n_sample:
        return df.sample(n_sample, random_state=42)
    return df


def load_amazon_dataset(
    return_pandas: bool = True,
    languages: Optional[List[str]] = None,
    use_stars: bool = False,
    n_sample: Optional[int] = None,
) -> Union[pd.DataFrame, datasets.Dataset]:
    """
    Load the train split of Amazon Reviews dataset from HuggingFace's datasets library.
    If return_pandas is True, return a pandas DataFrame. Otherwise, return
    a HuggingFace Dataset.
    """

    if languages is None:
        languages = ["en", "fr", "de", "es", "ja", "zh"]
    dataset = datasets.load_dataset(
        "amazon_reviews_multi", split="train", languages=languages
    )

    if return_pandas:
        df = _process_dataset_to_data_frame(dataset, use_stars=use_stars)
        if n_sample:
            return df.sample(n_sample, random_state=42)
        return df

    return dataset


def split_dataset(
    dataset: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test sets.
    """

    train, test = train_test_split(dataset, test_size=test_size, random_state=42)
    train, val = train_test_split(train, test_size=val_size, random_state=42)

    return train, val, test
