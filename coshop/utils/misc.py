"""Miscellaneous utilities shared across coshop.

Includes:

- :class:`Stopwatch` — wall-clock timer.
- :func:`parse_json` / :func:`parse_for_answer_tags` — LLM output parsers.
- :func:`print_debug` — verbosity-gated debug printer.
- :func:`strip_thinking_tokens` — removes ``<think>`` blocks from model output.
- JSON serialisation helper: :func:`_clean_for_json`.
"""

import torch
import numpy as np
from ast import literal_eval
import pickle
import json
import os
import re
from time import perf_counter
import pandas as pd
import hashlib
import zlib
from json_repair import repair_json
from PIL import Image

def strip_thinking_tokens(
    text: str,
    start_token: str = "<think>",
    end_token: str = "</think>",
) -> str:
    """Remove <think>...</think> blocks from text (e.g. before showing response to user)."""
    if text is None or not isinstance(text, str):
        return "" if text is None else text
    pattern = re.escape(start_token) + r".*?" + re.escape(end_token)
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def hash(x: object, type="md5"):
    """
    Hash an object.
    """
    # encode the object
    if isinstance(x, torch.Tensor):
        encoded = x.numpy().tobytes()
    elif isinstance(x, np.ndarray):
        encoded = x.tobytes()
    elif isinstance(x, str):
        encoded = x.encode("utf-8")
    elif isinstance(x, pd.DataFrame):
        encoded = pickle.dumps(x.to_dict(orient="records"))
    else:
        encoded = pickle.dumps(x)
    # hash the encoded object
    if type == "md5":
        return hashlib.md5(encoded).hexdigest()
    elif type == "sha256":
        return hashlib.sha256(encoded).hexdigest()
    else:
        return zlib.adler32(encoded)


def explode_df(df: pd.DataFrame, col_to_explode: str, drop_original: bool = True) -> pd.DataFrame:
    """
    Explode a column that contains an iterable (e.g. lists of strings or a dictionary)
    The new dataframe will have one column per element of the iterable

    For dicts, we only care about the keys.

    Example:
    df = pd.DataFrame({"tags": [["new", "2019"], ["old", "2020"]]})
    new_df = explode_df(df, "tags")
    print(new_df)
    # Output:
    #   new old 2019 2020
    # 0   True False True False
    # 1   False True False True

    If a tag has the same name as a column in the dataframe, we prioritize the column and ignore the tag.
    """
    if col_to_explode not in df.columns:
        return pd.DataFrame()
    series = df[col_to_explode]
    # Collect elements per row: for dicts use keys, else treat as iterable

    def _elements(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        try:
            val = eval(val)
        except Exception:
            pass
        if isinstance(val, dict):
            return [str(k) for k in val.keys()]
        elif isinstance(val, list):
            return [str(x) for x in val]
        elif isinstance(val, str):
            return [str(x) for x in val.split(",")]
    row_elements = [_elements(v) for v in series]
    all_elements = sorted({e for elems in row_elements for e in elems})
    if not all_elements:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(
        {e.strip(): [True if e.strip() in elems else False for elems in row_elements]
            for e in all_elements if ((e.strip() != "") and (e.strip() not in df.columns))},
        index=df.index,
    )

    # join to original df
    out = df.join(out)
    if drop_original:
        out = out.drop(columns=[col_to_explode])
    return out


class Stopwatch:
    """
    Context manager for timing a block of code
    Source: https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
    """

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.time = perf_counter() - self.time


def parse_json(json_str, return_start_end=False):
    """
    Parse a JSON string, returning None if it fails.
    If there are multiple JSONs in the string, returns the first one.
    If return_start_end is True, also returns a tuple of (start, end) of the JSON string.
    """
    if not isinstance(json_str, str):
        json_str = str(json_str)
    if json_str is None:
        if return_start_end:
            return None, None
        else:
            return None

    # replace occurrences of 2+ \ with a single \
    json_str = re.sub(r"\\{1,}n", r"\n", json_str)  # \\\\n -> \n
    json_str = re.sub(r"\\{1,}'", "'", json_str)  # \\\' -> \'
    json_str = re.sub(r'\\{1,}"', '"', json_str)  # \\\" -> \"

    if "```json" in json_str:
        start_end = (json_str.find("```json"), json_str.rfind("```") + 3)
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        start_end = (json_str.find("```"), json_str.rfind("```") + 3)
        json_str = json_str.split("```")[1].split("```")[0].strip()
    elif "{" in json_str and "}" in json_str:
        start_end = (json_str.find("{"), json_str.rfind("}") + 1)
        bracket_start_end = (json_str.find("["), json_str.rfind("]") + 1)
        if (bracket_start_end[0] != -1) and (bracket_start_end[0] < start_end[0]):
            start_end = bracket_start_end
        json_str = json_str[start_end[0] : start_end[1]]
    else:
        start_end = None

    json_str = repair_json(json_str)

    try:
        js = json.loads(json_str)
    except:
        try:
            js = literal_eval(json_str)
        except:
            try:
                import demjson3

                js = demjson3.decode(json_str)
            except:
                js = None

    if return_start_end:
        return js, start_end
    else:
        return js

def parse_for_answer_tags(
    text,
    keyword="answer",
    return_start_end=False,
    return_none_if_not_found=False,
    return_all=False,
):
    """
    Looks for <keyword>X</keyword> in text and returns X.
    """
    match = re.findall(rf"<{keyword}>(.*?)</{keyword}>", text, re.DOTALL)
    if match:
        if return_start_end:
            return match[0].strip(), (
                text.find(f"<{keyword}>"),
                text.find(f"</{keyword}>") + len(f"</{keyword}>"),
            )
        if return_all:
            return [item.strip() for item in match]
        else:
            return match[0].strip()
    else:
        if return_none_if_not_found:
            if return_start_end:
                return None, None
            return None
        else:
            if return_start_end:
                return text, (0, len(text))
            return text

# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ORANGE = "\033[38;5;216m"  # A softer, more pale orange
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    NONE = ""


def print_debug(
    message: str, function_name: str, color: str = "NONE", indent_level: int = 0
):
    """Print a debug message with function context and color.

    Args:
        message: The message to print (can be multiline)
        function_name: Name of the function generating the message
        color: Color to use for the function name (default: "NONE")
        indent_level: Number of indentation levels (default: 0)
    """
    message = str(message)
    color_code = getattr(Colors, color.upper(), Colors.NONE)
    indent = "\t" * indent_level

    # Split message into lines and indent each line
    lines = message.split("\n")
    indented_lines = [f"{indent}{line}" for line in lines]
    indented_message = "\n".join(indented_lines)

    print(f"{color_code}[{function_name}] {indented_message}{Colors.ENDC}")



def _clean_for_json(obj):
    """
    Recursively clean an object to ensure it's JSON serializable.

    Args:
        obj: Any object to clean

    Returns:
        JSON serializable version of the object
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    elif hasattr(obj, "isoformat"):
        # Handle datetime objects
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        # Handle custom objects by converting to dict
        return _clean_for_json(obj.__dict__)
    elif hasattr(obj, "tolist"):
        # Handle numpy arrays
        return obj.tolist()
    elif hasattr(obj, "item"):
        # Handle numpy scalars
        return obj.item()
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        # Handle other iterables (sets, etc.)
        return [_clean_for_json(item) for item in obj]
    else:
        # Convert everything else to string
        return str(obj)


def check_if_file_ending_exists(ending: str, output_dir: str) -> bool:
    """
    Check if a file of the format {output_dir}/*{ending} exists.
    """
    for file in os.listdir(output_dir):
        if file.endswith(ending):
            return True
    return False


def download_file_from_google_drive(
    file_id: str,
    output_path: str,
    unzip: bool = False,
    chunk_size: int = 8192,
    timeout: int = 600,
):
    """
    Download a file from Google Drive.
    """
    import gdown
    import zipfile

    os.makedirs(output_path, exist_ok=True)

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        result = gdown.download(url, output_path + f"/{file_id}.zip")
        if not result:
            # gdown returns None (without raising) when the file is not
            # publicly accessible. Treat that as a failure.
            raise RuntimeError(
                f"gdown could not download file {file_id!r}. Make sure the "
                "Drive file's sharing is set to 'Anyone with the link'."
            )
        print(f"File downloaded successfully to {output_path + f'/{file_id}.zip'}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

    if unzip:
        with zipfile.ZipFile(output_path + f"/{file_id}.zip", "r") as zip_ref:
            zip_ref.extractall(output_path)
        print(f"File unzipped successfully to {output_path}")
    return True

def check_na(value, allow_empty: bool = False):
    """
    Check if a value is NA (handles pd.NA, np.nan, None, and list containing any).
    """
    # Explicitly handle pd.NA (pandas NAType); pd.isna(pd.NA) is True but explicit check is clearer
    try:
        if value == float("nan") or value == np.nan:
            return True
        if str(value).lower() == "nan":
            return True
    except:
        pass
    if isinstance(value, list):
        if pd.isna(value).any():
            return True
        if allow_empty and len(value) == 0:
            return True
        return False
    elif pd.isna(value):
        return True
    elif allow_empty and value == "":
        return True
    return False


def parse_set(value, separator: str = ", "):
    """
    Parse a value into a set of strings for set-based comparison (e.g. Jaccard).
    - str: split on separator ("x, y, z" -> {"x", "y", "z"}).
      If str looks like a dict/list (starts with '{' or '['), eval and recurse.
    - dict: use keys as the set (e.g. {"tag_a": 1, "tag_b": 2} -> {"tag_a", "tag_b"}).
    - list/tuple: convert elements to strings.
    - None/NaN: return empty set.
    """
    if value is None or (isinstance(value, float) and __import__("math").isnan(value)):
        return set()
    try:
        import pandas as pd
        if pd.isna(value):
            return set()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k).strip() for k in value.keys() if str(k).strip()}
    if isinstance(value, (list, tuple)):
        return {str(x).strip() for x in value if str(x).strip()}
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return parse_set(eval(s), separator)
            except Exception:
                pass
        return {x.strip() for x in value.split(separator) if x.strip()}
    return {str(value).strip()} if str(value).strip() else set()

def is_same_image(image1: Image.Image, image2: Image.Image) -> bool:
    """
    Check if two images are the same.
    """
    return image1.tobytes() == image2.tobytes()