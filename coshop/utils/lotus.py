from lotus.models import LM
import lotus
from typing import Callable, Optional
import pandas as pd


def configure_lotus(model_name: str, model_kwargs: dict, max_tokens: int = 8000) -> LM:
    """
    Configure the Lotus LM to use the given model and model kwargs
    Args:
        model_name: The name of the model to use
        model_kwargs: The kwargs to pass to the model
            Handles vLLM models by looking for the model_provider and vllm_api_url keys inside model_kwargs
            - model_provider: The provider of the model (e.g. "vllm")
            - vllm_api_url: The URL of the vLLM API (e.g. "http://localhost:8000")
        max_tokens: The maximum number of tokens to use
    Returns:
        The Lotus LM configured to use the given model and model kwargs
    """
    lm_kwargs = model_kwargs.copy()
    provider = lm_kwargs.pop("model_provider", None)
    vllm_url = lm_kwargs.pop("vllm_api_url", None)
    lotus_model = model_name
    if provider == "vllm" and vllm_url:
        vllm_url = str(vllm_url).rstrip("/")
        lm_kwargs["api_base"] = (
            vllm_url if vllm_url.endswith("/v1") else vllm_url + "/v1"
        )
        if not lotus_model.startswith("hosted_vllm/"):
            lotus_model = f"hosted_vllm/{model_name}"
    lm = LM(model=lotus_model, max_tokens=max_tokens, **lm_kwargs)
    lotus.settings.configure(lm=lm)
    return lm


def sem_map_with_retries(df: pd.DataFrame, prompt: str, validation_fn: Callable[[str], bool] = None, retries: int = 3, **kwargs) -> pd.DataFrame:
    """
    Apply sem_map with retries to a dataframe.
    """
    if validation_fn is None:
        def validation_fn(x): return True

    output_values = df.copy()
    df_retry = df.copy()

    for _ in range(retries):
        if df_retry.empty:
            break

        try:
            df_retry = df_retry.sem_map(prompt, **kwargs)
        except Exception as e:
            print("Error when calling sem_map:", e)
            continue

        if "_map" not in df_retry.columns:
            continue

        # Apply validation fn to each output
        df_retry["_valid"] = df_retry["_map"].apply(validation_fn)
        valid_mask = df_retry["_valid"]

        # Update only the valid rows in the accumulated output
        if valid_mask.any():
            output_values.loc[df_retry.index[valid_mask],
                              "_map"] = df_retry.loc[valid_mask, "_map"]

        # Subset to invalid & retry
        df_retry = df_retry.loc[~valid_mask].drop(columns=["_valid"])

    # Check if we have a _map column
    if "_map" not in output_values.columns:
        raise ValueError("LOTUS sem_map did not produce the expected '_map' column.")

    return output_values


def sem_filter_with_retries(
    df: pd.DataFrame,
    user_instruction: str,
    validation_fn: Optional[Callable[[pd.DataFrame], bool]] = None,
    retries: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """
    Apply LOTUS sem_filter with retries.

    Calls DataFrame.sem_filter(user_instruction, **kwargs). On exception or
    when validation_fn returns False (if provided), retries up to `retries`
    times. Passes through optional sem_filter args (e.g. return_raw_outputs,
    default, suffix, examples, cascade_args, return_stats).

    Args:
        df: Input dataframe.
        user_instruction: Natural-language predicate for filtering (see LOTUS sem_filter).
        validation_fn: Optional callable that takes the filtered DataFrame and returns
            True to accept the result, False to retry.
        retries: Number of retry attempts.
        **kwargs: Additional keyword args forwarded to sem_filter.

    Returns:
        Filtered dataframe (subset of rows that pass the predicate).
    """
    if validation_fn is None:
        def validation_fn(_): return True

    last_exception = None
    result = pd.DataFrame()
    df_retry = df.copy()

    for _ in range(max(1, retries)):
        try:
            result = df_retry.sem_filter(user_instruction=user_instruction, **kwargs)
            if validation_fn(result):
                return result
        except Exception as e:
            last_exception = e
            continue

    if last_exception is not None:
        raise last_exception
    # Validation failed every time; return last result
    return result
