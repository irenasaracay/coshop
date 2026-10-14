"""Shared CLI and file helpers for the per-user asset generators."""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from .. import DATASETS, get_dataset
from ..dataset import ROOT_DIR, TEST_SET_SIZE, Dataset

# Config keys of the test-set asset files. Generators never write to these.
TEST_SET_ASSET_KEYS = ("sec_split", "initial_known_features", "baseline_queries")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=DATASETS + ["all"],
        default="all",
        help="Dataset to process (default: all).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        help="Dataset version (default: v2).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=TEST_SET_SIZE,
        help=f"First user index, inclusive (default: {TEST_SET_SIZE}, the first extended-split user).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last user index, exclusive (default: all users).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: the dataset's extended-split asset file). "
        "Only valid with a single --dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate users that already have an entry in the output file.",
    )


def resolve_datasets(args: argparse.Namespace) -> List[str]:
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    if args.out is not None and len(names) > 1:
        raise ValueError("--out requires a single --dataset")
    return names


def load_dataset(name: str, args: argparse.Namespace, **kwargs: Any) -> Dataset:
    """Load the dataset split that contains users ``[args.start, args.end)``."""
    if args.start < TEST_SET_SIZE < (args.end or float("inf")):
        raise ValueError(
            f"--start/--end must not straddle the test/extended boundary ({TEST_SET_SIZE})"
        )
    split = "test" if args.start < TEST_SET_SIZE else "extended"
    return get_dataset(name, version=args.version, max_xstar=1, split=split, **kwargs)


def user_indexes(ds: Dataset, args: argparse.Namespace) -> List[int]:
    end = ds.total_users if args.end is None else min(args.end, ds.total_users)
    return list(range(args.start, end))


def output_path(ds: Dataset, config_key: str, args: argparse.Namespace) -> str:
    """Resolve the output file, refusing to write to a test-set asset file."""
    assets_dir = ROOT_DIR / ds.dataset_name / "assets"
    if args.out is not None:
        path = os.path.abspath(args.out)
    else:
        filename = ds._config.get(config_key)
        if not filename:
            raise ValueError(
                f"{ds.dataset_name} {ds.version} config has no {config_key!r} entry"
            )
        path = str(assets_dir / filename)
    protected = {
        str(assets_dir / ds._config[k]) for k in TEST_SET_ASSET_KEYS if ds._config.get(k)
    }
    if path in protected:
        raise ValueError(f"Refusing to overwrite test-set asset file {path}")
    return path


def todo_indexes(
    indexes: List[int], existing: Dict[str, Any], args: argparse.Namespace, out: str
) -> List[int]:
    """Drop users that already have output (unless --overwrite).

    Users below TEST_SET_SIZE are refused for the default extended output.
    """
    if args.out is None and any(i < TEST_SET_SIZE for i in indexes):
        raise ValueError(
            f"Extended-split files only hold users >= {TEST_SET_SIZE}; pass --out to write elsewhere"
        )
    todo = [i for i in indexes if args.overwrite or str(i) not in existing]
    print(f"{out}: {len(indexes)} users requested, {len(todo)} to generate")
    return todo


def read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    ordered = {k: data[k] for k in sorted(data, key=int)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ordered, f, indent=2)
