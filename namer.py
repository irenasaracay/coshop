import datetime
from typing import Dict, Any
import os
import uuid
from utils.misc import check_if_file_ending_exists

ARGS = [
    "dataset",
    "spec_index",
]
KWARGS = [
    "policy",
    "policy_model",
    "simulator",
    "seed",
]
ARG_SEP = "_"
LIST_SEP = ","
SPACE = "-"


def _clean(s) -> str:
    """
    Clean the value
    Examples:
    - _clean(None) -> "None"
    - _clean("a") -> "a"
    - _clean(["a", "b", "c"]) -> "a,b,c"
    - _clean(1) -> "1"
    """
    if type(s) in (list, tuple):
        return LIST_SEP.join([_clean(x) for x in s])
    elif type(s) == str:
        return (
            s.replace(" ", SPACE)
            .replace(ARG_SEP, SPACE)
            .replace("=", SPACE)
            .replace("/", SPACE)
            .replace(LIST_SEP, SPACE)
        )
    elif s == None:
        return "None"
    elif type(s) in [int, float, bool]:
        return str(s)
    else:
        raise ValueError(f"Invalid type: {type(s)}")


def _fmt(k, v) -> str:
    """
    Format the key and value as a string
    Examples:
    - _fmt(None, "a") -> "_a"
    - _fmt("k", "v") -> "_k=v"
    - _fmt("k", ["a", "b", "c"]) -> "_k=a,b,c"
    - _fmt("k", {"a": "b", "c": "d"}) -> "_a=b_c=d"
    """
    kstr = "" if k is None else _clean(k) + "="
    if type(v) == dict:
        out = ""
        for vk, vv in v.items():
            out += f"{ARG_SEP}{_clean(vk)}={_clean(vv)}"
        return out
    else:
        return f"{ARG_SEP}{kstr}{_clean(v)}"


def get_experiment_name(
    fill_missing_with_star: bool = False, include_datetime: bool = False, **kwargs
) -> str:
    """
    Given the arguments and keyword arguments, return a unique experiment name
    """
    name = (
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "_"
        if include_datetime
        else ""
    )
    for arg in ARGS:
        if arg not in kwargs:
            if fill_missing_with_star:
                kwargs[arg] = "*"
            else:
                raise ValueError(f"Missing argument: {arg}")
        name += _fmt(None, kwargs[arg])
    for kwarg in KWARGS:
        if kwarg not in kwargs:
            if fill_missing_with_star:
                kwargs[kwarg] = "*"
            else:
                raise ValueError(f"Missing keyword argument: {kwarg}")
        name += _fmt(kwarg, kwargs[kwarg])
    return name.replace(ARG_SEP, "", 1)

def _unclean(s: str) -> Any:
    """
    Convert a cleaned string back to its original value
    Examples:
    - _unclean("None") -> None
    - _unclean("a") -> "a"
    - _unclean("a,b,c") -> ["a", "b", "c"]
    - _unclean("1") -> 1
    - _unclean("True") -> True
    """
    if s == "None":
        return None
    elif s == "True":
        return True
    elif s == "False":
        return False
    elif s.replace(".", "").isdigit():  # Handle both int and float
        return float(s) if "." in s else int(s)
    elif LIST_SEP in s:
        return [_unclean(x) for x in s.split(LIST_SEP)]
    else:
        return s

def get_args(path: str) -> Dict[str, Any]:
    """
    Given a path saved exactly as by get_experiment_name,
    get the arguments from the path
    """
    experiment_name = os.path.basename(path).rsplit(".", 1)[0]
    
    # Split into parts, removing any datetime prefix if present
    parts = experiment_name.split(ARG_SEP)
    if len(parts) > 0:
        try:
            # Try to parse the first part as a datetime
            datetime.datetime.strptime(parts[0], "%Y-%m-%d-%H-%M")
            parts = parts[1:]  # Remove datetime prefix if valid
        except ValueError:
            pass  # Not a datetime, keep the part
    
    args = {}
    
    # Process positional arguments first
    for i, arg in enumerate(ARGS):
        if i >= len(parts):
            break
        value = parts[i]
        if value == "*":
            continue
        args[arg] = _unclean(value)
    
    # Process keyword arguments
    for key in KWARGS:
        if f"{_clean(key)}=" in experiment_name:
            value = experiment_name.split(f"{_clean(key)}=")[1].split(ARG_SEP)[0]
            if value == "*":
                continue
            args[key] = _unclean(value) 

    return args


def get_checkpoint_file_path(output_path: str) -> str:
    """
    Generate a checkpoint file path in the same directory as the output path.

    Args:
        output_path: Path to the interaction JSON file

    Returns:
        Path to the checkpoint file with a UUID-based name
    """
    output_dir = os.path.dirname(output_path)
    checkpoint_filename = f"checkpoint_{uuid.uuid4()}.json"
    return os.path.join(output_dir, checkpoint_filename)


def check_mk_combination_exists(
    base_filename_no_datetime: str,
    m: int,
    m_processed: int | None,
    k: int,
    output_dir: str,
    mode: str | None = None,
    max_queries: int | None = None,
    max_execution_items: int | None = None,
) -> bool:
    """
    Check if a file exists for a specific execution_max_per_retrieval/k combination (and optionally mode, execution_max_queries, and execution_global_max).
    
    Args:
        base_filename_no_datetime: Base filename without datetime prefix
        m: Original execution_max_per_retrieval value (kept as 'm' for backward compatibility in function signature)
        m_processed: Processed execution_max_per_retrieval value (None if m was -1)
        k: k value
        output_dir: Directory to check in
        mode: Optional execution mode (e.g., "agentic" or "rank")
        max_queries: Optional execution_max_queries (kept as 'max_queries' for backward compatibility)
        max_execution_items: Optional execution_global_max (kept as 'max_execution_items' for backward compatibility)
        
    Returns:
        bool: True if file exists, False otherwise. If ``mode`` is ``"parser"`` and no
        ``_mode=parser`` file exists, also returns True when the same m/k/mq/gm file
        exists with ``_mode=agentic`` (parser reranks can be derived from agentic runs).
    """
    m_str = "None" if m_processed is None else str(m)
    filename_ending = f"{base_filename_no_datetime}_mi={m_str}_k={k}"
    if mode is not None:
        filename_ending += f"_mode={mode}"
    if max_queries is not None:
        filename_ending += f"_mq={max_queries}"
    if max_execution_items is not None:
        filename_ending += f"_gm={max_execution_items}"
    filename_ending += ".json"
    print("checking filename_ending: ", filename_ending)
    if check_if_file_ending_exists(filename_ending, output_dir):
        return True
    if mode == "parser":
        filename_agentic = f"{base_filename_no_datetime}_mi={m_str}_k={k}"
        filename_agentic += "_mode=agentic"
        if max_queries is not None:
            filename_agentic += f"_mq={max_queries}"
        if max_execution_items is not None:
            filename_agentic += f"_gm={max_execution_items}"
        filename_agentic += ".json"
        print("checking filename_ending (parser fallback -> agentic): ", filename_agentic)
        return check_if_file_ending_exists(filename_agentic, output_dir)
    return False


def check_all_mk_combinations_exist(
    base_filename_no_datetime: str,
    m_values: list,
    m_values_processed: list,
    k_values: list,
    output_dir: str,
    mode: str | None = None,
    max_queries: int | None = None,
    max_execution_items: int | None = None,
) -> bool:
    """
    Check if files exist for all m/k combinations (and optionally mode, max_queries, max_execution_items).
    
    Args:
        base_filename_no_datetime: Base filename without datetime prefix
        m_values: List of m values
        m_values_processed: List of processed m values (None for -1)
        k_values: List of k values
        output_dir: Directory to check in
        mode: Optional execution mode (e.g., "agentic" or "rank"). Can be a list for multiple modes.
        max_queries: Optional execution_max_queries. Can be a list for multiple values.
        max_execution_items: Optional execution_global_max. Can be a list for multiple values.
        
    Returns:
        bool: True if all combinations exist, False otherwise. For ``mode="parser"``,
        see :func:`check_mk_combination_exists` (agentic satisfies parser when parser
        file is absent).
    """
    # Normalize mode, max_queries, and max_execution_items to lists
    if mode is None:
        modes = [None]
    elif not isinstance(mode, list):
        modes = [mode]
    else:
        modes = mode
    
    if max_queries is None:
        max_queries_list = [None]
    elif not isinstance(max_queries, list):
        max_queries_list = [max_queries]
    else:
        max_queries_list = max_queries
    
    if max_execution_items is None:
        max_execution_items_list = [None]
    elif not isinstance(max_execution_items, list):
        max_execution_items_list = [max_execution_items]
    else:
        max_execution_items_list = max_execution_items
    
    # Check all combinations
    for m, m_processed in zip(m_values, m_values_processed):
        for k in k_values:
            for execution_mode in modes:
                for max_q in max_queries_list:
                    for max_items in max_execution_items_list:
                        if not check_mk_combination_exists(
                            base_filename_no_datetime,
                            m,
                            m_processed,
                            k,
                            output_dir,
                            mode=execution_mode,
                            max_queries=max_q,
                            max_execution_items=max_items,
                        ):
                            return False
    return True
    


def get_condition(interaction):
    """
    Determine the experimental condition from interaction configuration.
    
    Args:
        interaction: ElicitationInteraction object
        
    Returns:
        str: The condition name (e.g., "full_knowledge", "grounded_raw-llm", "ungrounded_raw-llm_catalog_corrupt")
    """
    config = interaction.config
    policy = interaction.policy
    
    if config.get('simulator') == "full_spec_user":
        return "full_knowledge"
    elif policy == "random_baseline":
        return "random_baseline"
    elif policy == "popularity_baseline":
        return "popularity_baseline"
    elif policy == "ustar_baseline":
        return "oracle"
    else:
        condition = "grounded" if config.get('retrieval_access') else "ungrounded"
        condition += "_" + policy
        if config.get("catalog_access"):
            condition += "_catalog"
        if config.get("corrupt_representations"):
            condition += "_corrupt"
        return condition

