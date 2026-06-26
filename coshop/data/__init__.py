"""Dataset loading utilities for the coshop benchmark.

The three supported datasets are ``"hm"`` (H&M fashion), ``"movielens"``
(movie ratings), and ``"goodreads"`` (book reviews).  Each dataset exposes a
catalog of items and a collection of :class:`~coshop.data.dataset.Specification`
objects, one per benchmark task.

Main exports:
    DATASETS: List of supported dataset name strings.
    get_dataset: Factory that returns a fully-initialised :class:`~coshop.data.dataset.Dataset`.
    get_spec: Convenience wrapper that returns a single :class:`~coshop.data.dataset.Specification`.
"""

from .dataset import Specification, Dataset

SUPPORTED_VERSIONS = ("v1", "v2")
DEFAULT_VERSION = "v2"


def load_dataset_config(dataset_root: str, version: str) -> dict:
    """Load a per-dataset YAML config file (e.g. v2_config.yml)."""
    import os
    import yaml
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported dataset version: {version!r}. Must be one of {SUPPORTED_VERSIONS}."
        )
    path = os.path.join(dataset_root, f"{version}_config.yml")
    with open(path, "r") as f:
        return yaml.safe_load(f)

__all__ = ["DATASETS", "get_dataset", "get_spec"]

DATASETS = [
    "hm",
    "movielens",
    "goodreads",
]


def get_dataset(dataset_name: str, **kwargs) -> "Dataset":
    """Return a dataset instance for the given dataset name.

    Datasets are imported lazily so that optional dependencies (e.g. heavy
    model weights) are only loaded when the requested dataset is actually used.

    Args:
        dataset_name: One of ``"hm"``, ``"movielens"``, or ``"goodreads"``.
        **kwargs: Forwarded verbatim to the dataset constructor
            (:class:`~coshop.data.hm.data.HMDataset`,
            :class:`~coshop.data.movielens.data.MovieLensDataset`, or
            :class:`~coshop.data.goodreads.data.GoodreadsDataset`).
            See those classes for the full list of accepted keyword arguments.

    Returns:
        An initialised :class:`~coshop.data.dataset.Dataset` subclass.

    Raises:
        ValueError: If ``dataset_name`` is not one of the supported values.
    """
    if dataset_name == "hm":
        from .hm.data import HMDataset
        return HMDataset(**kwargs)
    elif dataset_name == "movielens":
        from .movielens.data import MovieLensDataset
        return MovieLensDataset(**kwargs)
    elif dataset_name == "goodreads":
        from .goodreads.data import GoodreadsDataset
        return GoodreadsDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_spec(dataset_name: str, index: int, **kwargs) -> "Specification":
    """Return a single benchmark specification by dataset name and index.

    Convenience wrapper around :func:`get_dataset` that loads the full dataset
    and immediately retrieves one :class:`~coshop.data.dataset.Specification`.

    Args:
        dataset_name: One of ``"hm"``, ``"movielens"``, or ``"goodreads"``.
        index: Zero-based index of the specification within the dataset.
        **kwargs: Forwarded to :func:`get_dataset`.

    Returns:
        The :class:`~coshop.data.dataset.Specification` at position ``index``.
    """
    print("Getting spec for", dataset_name, index, kwargs)
    dataset = get_dataset(dataset_name, **kwargs)
    return dataset[index]
