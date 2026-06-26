"""Model wrappers for LLM and embedding APIs used across coshop.

Main exports:

- :class:`LangChainModel` — wraps a LangChain chat model with tool-calling
  support and a ``copy_for_prediction`` helper.
- :class:`EmbeddingModelWrapper` — wraps a ``SentenceTransformer`` or a remote
  embedding API with disk caching.
- Helper functions: :func:`get_token_usage`, :func:`is_openai_model`,
  :func:`is_anthropic_model`, :func:`is_gemini_model`.
"""

from typing import List, Tuple, Union, Any, Optional, Dict
import json
import os
import re
import hashlib
import uuid
from pathlib import Path
from PIL import Image
import base64

from functools import lru_cache
import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def encode_image_as_user_msg(
    image: Image.Image = None,
    image_path: str = None,
    extension: str = "png",
    caption: str = None,
    model_name: str = "gpt-5-nano",
) -> str:
    """
    Encode an image to a base64 string.
    """
    if image_path is None:
        assert image is not None
        need_to_remove = True
        image.save(f"temp.{extension}")
        image_path = f"temp.{extension}"
    else:
        need_to_remove = False

    with open(image_path, "rb") as image_file:
        bytes = base64.b64encode(image_file.read()).decode("utf-8")
        content = [] if caption is None else [{"type": "text", "text": caption}]
        msg = {
            "role": "user",
            "content": content
            + [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{extension};base64,{bytes}"},
                },
            ],
        }

    if need_to_remove:
        os.remove(image_path)
    return msg


class Model:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def fmt_as_dialog(
        self,
        prompts: List[Union[str, Image.Image]] = None,
        dialogs: List[List[Tuple[str, Union[str, Image.Image]]]] = None,
    ):
        """
        Args:
            prompts: A list of prompts. The list dimension is over (B,)
                e.g. ["What is the capital of France?"] -> [[{"role": "user", "content": "What is the capital of France?"}]]
            dialogs: A list of [(role, content)] pairs. The list dimension is over (B, D)
                e.g. [[("user", "What is the capital of France?"), ("assistant", "Paris.")]] -> [[{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}]]
        Returns:
            Formatted dialogs: list of list of dictionaries (B, D) where each dictionary has keys "role" and "content"
        """
        assert (prompts is None) ^ (dialogs is None), (
            "Exactly one of prompts or dialogs must be provided."
        )
        out = []
        if prompts is not None:
            for prompt in prompts:
                if prompt is None or len(prompt) == 0:
                    continue
                if isinstance(prompt, Image.Image):
                    out.append(encode_image_as_user_msg(image=prompt))
                else:
                    out.append([{"role": "user", "content": prompt}])
        else:
            for dialog in dialogs:
                o = []
                for role, content in dialog:
                    if content is None or len(content) == 0:
                        continue
                    if isinstance(content, Image.Image):
                        o.append(encode_image_as_user_msg(image=content))
                    else:
                        o.append({"role": role, "content": content})
                out.append(o)
        return out

    def generate(
        self,
        *,
        prompts: List[str] = None,
        dialogs: List[List[Tuple[str, str]]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Args:
            prompts: A list of prompts. The list dimension is over (B,)
                e.g. ["What is the capital of France?"]
            dialogs: A list of [(role, content)] pairs. The list dimension is over (B, D)
                e.g. [[("user", "What is the capital of France?"), ("assistant", "Paris.")]]
        Returns:
            A list of completions. The list dimension is over (B,)
            e.g. ["Paris."]
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


def get_reasoning_effort_kwargs(
    model_name: str, reasoning_effort: str, max_tokens=64000
) -> dict:
    """
    Get the kwargs for the reasoning effort for a model.
    """
    if any(
        model_name.lower().startswith(m)
        for m in ["gpt-5", "o3", "o4", "o1", "openai", "qwen"]
    ):
        if reasoning_effort == "minimal":
            reasoning_effort = "low"
        return {"reasoning_effort": reasoning_effort}
    elif is_anthropic_model(model_name):
        print("ANTHROPIC MODEL: ", model_name)
        if "claude-sonnet-4-6" in model_name:
            if reasoning_effort == "minimal":
                reasoning_effort = "low"
            return {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": reasoning_effort},
            }

        # legacy models can disable thinking
        if reasoning_effort == "minimal":
            return {"thinking": {"type": "disabled"}}
        effort_to_tokens = {
            "low": 0.2 * max_tokens,
            "medium": 0.4 * max_tokens,
            "high": 0.8 * max_tokens,
        }
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": max(
                    1024, int(effort_to_tokens.get(reasoning_effort, max_tokens))
                ),
            }
        }
    elif is_gemini_model(model_name):
        effort_to_tokens = {
            "minimal": 0,
            "low": 0.2 * max_tokens,
            "medium": 0.4 * max_tokens,
            "high": 0.8 * max_tokens,
        }
        if model_name == "gemini-2.5-pro" and reasoning_effort == "minimal":
            print("Warning: Gemini 2.5 Pro does not support minimal reasoning effort")
            reasoning_effort = "low"
        return {
            "thinking_budget": int(effort_to_tokens.get(reasoning_effort, max_tokens))
        }
    print("Warning: model not supported for reasoning effort: ", model_name)
    return {}


@lru_cache(maxsize=10)
def is_openai_model(model_name: str, api_key: str = None) -> bool:
    """
    Check if a model is an OpenAI model.
    Args:
        model_name: The name of the model to check.
        api_key: Optional API key for OpenAI client.
    Returns:
        True if the model is an OpenAI model, False otherwise.
    """
    return model_name.startswith(("gpt-", "o1", "o3", "openai/"))

    from openai import OpenAI

    try:
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        return model_name in [m.id for m in client.models.list()]
    except Exception:
        # If we can't connect to OpenAI API, fall back to name-based detection
        return model_name.startswith(("gpt-", "o1", "o3"))


@lru_cache(maxsize=10)
def is_anthropic_model(model_name: str, api_key: str = None) -> bool:
    """
    Check if a model is an Anthropic model.
    Args:
        model_name: The name of the model to check.
        api_key: Optional API key for Anthropic client.
    Returns:
        True if the model is an Anthropic model, False otherwise.
    """
    from anthropic import Anthropic

    try:
        client = Anthropic(api_key=api_key) if api_key else Anthropic()
        return model_name in [m.id for m in client.models.list()]
    except Exception:
        # If we can't connect to Anthropic API, fall back to name-based detection
        return model_name.startswith("claude")


def is_gemini_model(model_name: str, api_key: str = None) -> bool:
    return "gemini" in model_name.lower()



def init_langchain_model(model_name: str, **kwargs):
    """
    Initialize a LangChain chat model.
    Args:
        model_name: The name of the model to initialize.
        **kwargs: Additional keyword arguments to pass to the model.
    Returns:
        A LangChain model.
    """
    from langchain.chat_models import init_chat_model

    #### Process kwargs ####

    # remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    # torch_dtype and quantization are only for HuggingFace; pop so not passed to OpenAI/Anthropic
    torch_dtype = kwargs.pop("torch_dtype", None)
    quantization = kwargs.pop("quantization", None)
    api_key = kwargs.get("api_key")
    if "reasoning_effort" in kwargs:
        kwargs.update(
            get_reasoning_effort_kwargs(
                model_name,
                kwargs.pop("reasoning_effort"),
                max_tokens=kwargs.get("max_tokens", 64000),
            )
        )

    # Gemini doesn't take seed
    if is_gemini_model(model_name):
        kwargs.pop("seed", None)


    #### Extract provider ####
    provider = kwargs.pop("model_provider", None)
    if provider is None:
        provider = (
            "openai"
            if is_openai_model(model_name, api_key=api_key)
            else (
                "anthropic"
                if is_anthropic_model(model_name, api_key=api_key)
                else (
                    "google_genai"
                    if is_gemini_model(model_name, api_key=api_key)
                    else "huggingface"
                )
            )
        )
    if provider == "vllm":
        provider = "openai"
        vllm_url = kwargs.pop("vllm_api_url", None) or os.environ.get("VLLM_API_URL")
        if not vllm_url:
            raise ValueError(
                "vllm requires vllm_api_url in model_kwargs or VLLM_API_URL in the environment"
            )
        kwargs["base_url"] = (
            vllm_url
            if str(vllm_url).endswith("/v1")
            else str(vllm_url).rstrip("/") + "/v1"
        )

    #### Initialize model ####
    if provider == "huggingface":
        # init_chat_model doesn't support HF correctly: https://github.com/langchain-ai/langchain/issues/28226
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline,
            BitsAndBytesConfig,
        )
        from langchain_huggingface import HuggingFacePipeline
        from .tools import ChatHuggingFaceTools

        # Resolve torch_dtype for faster inference (e.g. bf16) when loading the model
        _dtype = torch_dtype
        if _dtype is not None and isinstance(_dtype, str):
            _dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            _dtype = _dtype_map.get(_dtype.lower(), getattr(torch, _dtype, _dtype))
        from_pretrained_kwargs = {}
        if quantization is not None and int(quantization) in (4, 8):
            q = int(quantization)
            from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=(q == 8),
                load_in_4bit=(q == 4),
            )
        elif _dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = _dtype

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **from_pretrained_kwargs
        )
        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens", 64000)

        print("Initializing pipeline with kwargs: ", kwargs)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            **kwargs,
        )
        hf = HuggingFacePipeline(pipeline=pipe)
        return ChatHuggingFaceTools(llm=hf, verbose=True)
    else:
        # Prepare kwargs for LangChain init_chat_model
        langchain_kwargs = kwargs.copy()

        # Map our API key parameter to LangChain's expected format
        if provider in ["openai", "anthropic"] and api_key:
            langchain_kwargs["api_key"] = api_key

        if "max_tokens" not in langchain_kwargs:
            langchain_kwargs["max_tokens"] = (
                None if provider != "anthropic" else 64000
            )  # anthropic needs max tokens

        return init_chat_model(
            model_name,
            model_provider=provider,
            **langchain_kwargs,
        )


# Prefix used to tag summarized (compressed) messages. Used both when writing the tag
# in _summarize_messages and when detecting already-compressed messages, so the two
# can never drift apart.
_SUMMARY_TAG_PREFIX = "[SUMMARY OF"

# Soft cap on summarizer output length (words). Keeps compressed tool results
# small so they shrink the policy's context as much as possible.
_SUMMARY_MAX_WORDS = 50

# System prompt for the summarizer. `{query_context}` is filled with the tool
# call(s) that produced the result (so the summarizer knows what was looked for),
# or "" when that context is unavailable.
_SUMMARY_SYSTEM_PROMPT = (
    "You are compressing a single tool/search result to save space. "
    f"Summarize the tool result below in fewer than {_SUMMARY_MAX_WORDS} words. "
    "Be concise: prefer terse phrasing and omit filler; every word should carry information. "
    "Write as if describing what was found — not as if describing the act of summarizing. "
    "Preserve specific item IDs, key results, and details useful for continuing the task. "
    "If you have enough space, try to preserve as many features as possible, "
    f"but remember your word limit of {_SUMMARY_MAX_WORDS} words. "
    "Keep what is relevant to the tool call (the query) shown below; "
    "you may drop content clearly irrelevant to it. "
    "If results were browsed, summarize trends. Do not mention that this is a summary.\n\n"
    "{query_context}"
)



class LangChainModel(Model):
    def __init__(
        self,
        model_name: str,
        model_no_tools=None,
        summary_model_name: str = None,
        tools: List[Any] = None,
        verbosity: int = 0,
        max_react_steps: int = 25,
        min_react_steps: int = 1,
        prompt_cache: bool = True,
        multiturn_memory: bool = False,
        summarize_state_after: int = None,
        out_of_steps_msg: str = None,
        list_tools_in_prompt: bool = False,
        thinking_tokens: Tuple[str, str] = ("<think>", "</think>"),
        add_thinking_tag: bool = True,
        empty_message_filler: Optional[str] = None,
        api_key: str = None,
        model_provider: str = None,
        summary_api_key: str = None,
        summary_model_provider: str = None,
        summary_vllm_api_url: str = None,
        **kwargs,
    ):
        super().__init__(model_name)
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
        from langchain.tools.render import render_text_description
        from langchain_core.messages import SystemMessage

        assert max_react_steps > 0, "max_react_steps must be positive"
        assert min_react_steps <= max_react_steps, (
            "min_react_steps must be less than or equal to max_react_steps"
        )
        self._init_kwargs = dict(
            kwargs
        )  # for copy_for_prediction; passed to init_langchain_model when model_no_tools is None

        if model_no_tools is None:
            self.model_no_tools = init_langchain_model(
                model_name, api_key=api_key, model_provider=model_provider, **kwargs
            )
        else:
            self.model_no_tools = model_no_tools

        if summary_model_name is None:
            self.summary_model = self.model_no_tools
        else:
            print(
                f"Using separate summary model {summary_model_name} with provider {summary_model_provider} {summary_vllm_api_url}"
            )
            self.summary_model = init_langchain_model(
                summary_model_name,
                api_key=summary_api_key,
                model_provider=summary_model_provider,
                vllm_api_url=summary_vllm_api_url,
            )

        if is_anthropic_model(model_name, api_key=api_key):
            bind_tools_kwargs = {
                "tool_choice": {
                    "type": "auto",
                    "disable_parallel_tool_use": True,
                }
            }
        elif is_openai_model(model_name, api_key=api_key):
            bind_tools_kwargs = {
                "parallel_tool_calls": False,
            }
        elif is_gemini_model(model_name, api_key=api_key):
            bind_tools_kwargs = {
                "tool_config": {
                    "function_calling_config": {
                        "mode": "AUTO",
                    }
                }
            }
        else:
            # hf model
            if not list_tools_in_prompt:
                print("Warning: list_tools_in_prompt is False for huggingface model")
            bind_tools_kwargs = {}

            # multiturn memory must be true if prompt_cache is true
            if prompt_cache:
                assert multiturn_memory, (
                    "For HF models, multiturn_memory must be true if prompt_cache is true"
                )

        # setup HERMES prompt to explain what tools are available
        if list_tools_in_prompt and tools is not None and len(tools) > 0:
            from .tools import post_model_parse_tools_hook, HERMES_PROMPT

            # This is the Hermes prompt
            self._system_tools_msg = HERMES_PROMPT.format(
                tools=render_text_description(tools),
                tool_names=", ".join([t.name for t in tools]),
            )
            if model_provider != "vllm":
                extra_kwargs = {"post_model_hook": post_model_parse_tools_hook}
            else:
                # VLLM handles its own tool parsing
                extra_kwargs = {}
        else:
            self._system_tools_msg = None
            extra_kwargs = {}

        # Per-step mid-rollout compression: shrink the LLM input (not stored state) once
        # enough fresh tool results accumulate. Gated on self._summarize_state_after, which
        # is None for normal agents (hook is then a pure no-op). Attributes it reads are set
        # just below; the hook only runs at invoke time, so ordering here is fine.
        self._summarize_state_after = summarize_state_after
        self._tool_summary_cache = {}

        # Cumulative token usage for EVERY LLM call this model makes: react-loop
        # generations (generate), summarizer/compression calls, and any other
        # model_no_tools.invoke. Counting at this (model) layer means the agent
        # never has to parse messages for usage, and prediction/report — which
        # run on a throwaway copy_for_prediction agent that SHARES this dict (see
        # copy_for_prediction) — are counted automatically. The agent reads this
        # directly via its cumulative_token_breakdown property.
        self.cumulative_token_breakdown: Dict[str, int] = {
            "input": 0,
            "input_cached": 0,
            "output": 0,
            "reasoning": 0,
        }
        # Output(+reasoning) token cost of the most recent generate() call, used
        # by the agent layer as the per-turn token_cost for budget tracking.
        self.last_call_token_cost: int = 0

        self.graph = create_react_agent(
            (
                self.model_no_tools.bind_tools(tools, **bind_tools_kwargs)
                if tools
                else self.model_no_tools
            ),
            tools=tools if tools is not None else [],
            debug=(verbosity == 2),
            checkpointer=MemorySaver(),  # need to keep this for state fetching
            pre_model_hook=self._pre_model_compress_hook,
            **extra_kwargs,
        )
        self._raw_state = []
        self._max_react_steps = max_react_steps
        self._min_react_steps = min_react_steps
        self._multiturn_memory = multiturn_memory
        self._out_of_steps_msg = out_of_steps_msg
        self._pre_compress_hook = None
        self._thinking_tokens = thinking_tokens
        self._add_thinking_tag = add_thinking_tag
        self._tools = tools
        self._empty_message_filler = empty_message_filler

        # whether to change system-only prompts to user-only prompts
        self._is_anthropic = is_anthropic_model(model_name, api_key=api_key)
        self._is_openai = is_openai_model(model_name, api_key=api_key)
        self._is_gemini = is_gemini_model(model_name)
        self._is_hf = not (self._is_anthropic or self._is_openai or self._is_gemini)
        self._prompt_cache = prompt_cache
        self._model_provider = model_provider
        self._api_key = api_key
        self._verbosity = verbosity
        self._list_tools_in_prompt = list_tools_in_prompt
        self.thread_id = str(uuid.uuid4())

        # Manually inject system tools message into state
        if self._system_tools_msg is not None:
            self.graph.update_state(
                {"configurable": {"thread_id": self.thread_id}},
                {"messages": [SystemMessage(content=self._system_tools_msg)]},
            )

    def _record_usage(self, messages: List[Any], update_last_call: bool) -> int:
        """
        Accumulate per-message token usage (input / input_cached / output /
        reasoning) for every AIMessage in `messages` into
        cumulative_token_breakdown, and return the summed output(+reasoning)
        token cost. When update_last_call is True (a top-level generate()), the
        returned cost is also stored as last_call_token_cost for the agent's
        per-turn budget accounting; auxiliary calls (e.g. summarization) pass
        False so they are counted in the breakdown but do not perturb the
        per-turn cost.
        """
        from langchain_core.messages import AIMessage

        call_cost = 0
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            call_cost += get_token_usage(msg.response_metadata, default_value=0)
            for k, v in get_token_breakdown(msg).items():
                self.cumulative_token_breakdown[k] += v
        call_cost = int(call_cost)
        if update_last_call:
            self.last_call_token_cost = call_cost
        return call_cost

    def reset_token_tracking(self) -> None:
        """Zero the cumulative token counters (in place, to preserve any shared
        references held by prediction-time agent copies)."""
        for k in self.cumulative_token_breakdown:
            self.cumulative_token_breakdown[k] = 0
        self.last_call_token_cost = 0

    def copy_for_prediction(
        self,
        *,
        tools: List[Any],
        min_react_steps: int = None,
        max_react_steps: int = None,
    ) -> "LangChainModel":
        """
        Create a copy of this agent with different tools and/or react step limits.
        Preserves all other constructor arguments (model_provider, api_key, etc.).
        Note this new agent has an empty state.
        For prediction-time agents (e.g., agentic ranking or final recommendations),
        we force temperature=0.0 for more deterministic behavior, regardless of the
        interactive agent's temperature.
        """
        # Start from the original model init kwargs, but override temperature to 0.0
        # so that prediction-time calls are deterministic.
        init_kwargs = dict(self._init_kwargs)

        prediction_agent = LangChainModel(
            model_name=self.name,
            model_no_tools=None,  # Re-init a fresh chat model with updated kwargs
            tools=tools,
            verbosity=self._verbosity,
            max_react_steps=(
                max_react_steps
                if max_react_steps is not None
                else self._max_react_steps
            ),
            min_react_steps=(
                min_react_steps
                if min_react_steps is not None
                else self._min_react_steps
            ),
            prompt_cache=self._prompt_cache,
            multiturn_memory=self._multiturn_memory,
            summarize_state_after=self._summarize_state_after,
            out_of_steps_msg=self._out_of_steps_msg,
            list_tools_in_prompt=self._list_tools_in_prompt,
            thinking_tokens=self._thinking_tokens,
            add_thinking_tag=self._add_thinking_tag,
            empty_message_filler=self._empty_message_filler,
            api_key=self._api_key,
            model_provider=self._model_provider,
            **init_kwargs,
        )
        # Share the token-usage sink so EVERYTHING the prediction agent does
        # (its react-loop generations, compress_state on creation, and per-step
        # mid-rollout compression) accumulates straight into this (the original)
        # agent's counter. The agent layer reads it via cumulative_token_breakdown.
        prediction_agent.cumulative_token_breakdown = self.cumulative_token_breakdown
        return prediction_agent

    def fmt_as_dialog(
        self,
        prompts: List[str] = None,
        dialogs: List[List[Tuple[str, str]]] = None,
        is_first_turn: bool = False,
    ):
        dialogs = super().fmt_as_dialog(prompts=prompts, dialogs=dialogs)
        for dialog in dialogs:
            for ix, msg in enumerate(dialog):
                if self._prompt_cache and self._is_openai:
                    msg["prompt_cache_key"] = self.thread_id
                # only openai allows non-first-turn system messages
                if not self._is_openai and not is_first_turn:
                    if msg["role"] == "system":
                        msg["role"] = "user"
                # Anthropic prompt caching: cache_control must live INSIDE a
                # content block — a top-level message key is silently dropped
                # into additional_kwargs and ignored by langchain_anthropic. We
                # mark the system prompt (the large, byte-stable prefix), which
                # caches tools + system together. Only system messages are
                # marked, so the breakpoint never accumulates across turns and
                # we stay well under Anthropic's 4-breakpoint-per-request cap.
                if (
                    self._prompt_cache
                    and self._is_anthropic
                    and msg.get("role") == "system"
                    and isinstance(msg.get("content"), str)
                ):
                    msg["content"] = [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
        return dialogs

    def generate(
        self,
        *,
        prompts: List[str] = None,
        dialogs: List[List[Tuple[str, str]]] = None,
        raw_messages: List[List[Any]] = None,
        remove_thinking_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        from langchain_core.messages import SystemMessage

        any_system_msg_in_state = any(
            isinstance(msg, SystemMessage) for msg in self.state
        )

        if prompts is not None:
            dialogs = self.fmt_as_dialog(
                prompts=prompts, is_first_turn=not any_system_msg_in_state
            )
        elif dialogs is not None:
            dialogs = self.fmt_as_dialog(
                dialogs=dialogs, is_first_turn=not any_system_msg_in_state
            )
        else:
            dialogs = raw_messages

        assert len(dialogs) == 1, (
            "Only one prompt / conversation at a time is supported"
        )

        # NOTE: proactive compression is driven per-step by the pre_model_hook
        # (see _pre_model_compress_hook), gated on self._summarize_state_after.

        outs = []
        for dialog in dialogs:
            outs.append(
                self._call_graph(
                    dialog,
                    **kwargs,
                )
            )

        # Count token usage for this generation (only one dialog is supported).
        self._record_usage(outs[0] if outs else [], update_last_call=True)

        if remove_thinking_tokens:
            pattern = (
                re.escape(self._thinking_tokens[0])
                + r".*?"
                + re.escape(self._thinking_tokens[1])
            )
            for out in outs:
                for msg in out:
                    msg.content = re.sub(pattern, "", msg.content, flags=re.DOTALL)

        return outs

    @property
    def state(self):
        """Return the state of the chain"""
        return self.graph.get_state(
            {"configurable": {"thread_id": self.thread_id}}
        ).values.get("messages", [])

    @property
    def raw_state(self):
        """
        Return the raw state of the chain (read-only)
        The raw-state only appends: no summarization, no deletion
        The exception is if multiturn_memory is False: then the raw_state
        will be [].
        """
        return self._raw_state

    def get_state(self, slice=None) -> str:
        """Dump the state of the chain"""
        from langchain_core.load import dumps

        if slice is None:
            state = self.state
        else:
            state = self.state[slice]
        return dumps(state)

    def load_state(self, state: str):
        """Load the state of the chain"""
        import json as _json
        from langchain_core.load import loads
        from langchain_core.messages import ToolMessage

        messages = loads(state)
        # OpenAI requires ToolMessage content to be a string, not a list.
        # When a tool returns [] (empty list), LangChain stores it as-is,
        # which causes a 400 validation error on the next API call.
        for msg in messages:
            if isinstance(msg, ToolMessage) and not isinstance(msg.content, str):
                msg.content = _json.dumps(msg.content)
        self.clear_state()
        self.graph.update_state(
            {"configurable": {"thread_id": self.thread_id}},
            {"messages": messages},
        )

    def mark_history_cache_breakpoint(self) -> bool:
        """
        Place an Anthropic prompt-cache breakpoint on the last message of the
        current state, so the whole conversation-history prefix is cached and
        re-read across subsequent generate() calls that share this history
        (e.g. prediction retries, the foregone-recall continuation, and the
        report/rank passes).

        cache_control must live inside a content block, so a string-content
        message is converted to a single text block carrying the breakpoint,
        and a list-content message gets the breakpoint attached to its last
        text block. Returns False (no-op) for non-Anthropic models, when prompt
        caching is disabled, when the state is empty, or when no suitable text
        block is found. Stays within Anthropic's 4-breakpoint cap: it marks a
        single message and never accumulates across turns.
        """
        if not (self._is_anthropic and self._prompt_cache):
            return False
        state = self.state
        if not state:
            return False
        last = state[-1]
        content = last.content
        if isinstance(content, str):
            if content.strip() == "":
                return False
            new_content = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list):
            new_content = [dict(b) if isinstance(b, dict) else b for b in content]
            text_ix = next(
                (
                    i
                    for i in range(len(new_content) - 1, -1, -1)
                    if isinstance(new_content[i], dict)
                    and new_content[i].get("type") == "text"
                ),
                None,
            )
            if text_ix is None:
                return False
            new_content[text_ix] = {
                **new_content[text_ix],
                "cache_control": {"type": "ephemeral"},
            }
        else:
            return False
        self._update_messages(ids=[last.id], content=[new_content])
        return True

    def __len__(self):
        """Return the length of the chain of messages"""
        return len(self.state)

    def clear_state(self):
        """Clear the state of the chain"""
        self.graph.checkpointer.delete_thread(self.thread_id)

    def _call_model_no_tools(
        self, messages: List[Union[dict, Any]], persist_state: bool = True
    ) -> Any:
        """Call the model without tools. This is guaranteed to return a single message."""
        msg = self.model_no_tools.invoke(messages)
        # Count usage (summarization / compression and any other no-tools call)
        # in the cumulative breakdown, but don't disturb the per-turn cost.
        self._record_usage([msg], update_last_call=False)
        if persist_state:
            self.graph.update_state(
                {"configurable": {"thread_id": self.thread_id}},
                {"messages": [msg]},
            )
        return msg

    def _call_summary_model(
        self, messages: List[Union[dict, Any]], persist_state: bool = False
    ) -> Any:
        """Call the model without tools. This is guaranteed to return a single message."""
        msg = self.summary_model.invoke(messages)
        # Count usage (summarization / compression and any other no-tools call)
        # in the cumulative breakdown, but don't disturb the per-turn cost.
        if self.summary_model == self.model_no_tools:
            self._record_usage([msg], update_last_call=False)
        if persist_state:
            self.graph.update_state(
                {"configurable": {"thread_id": self.thread_id}},
                {"messages": [msg]},
            )
        return msg

    def _summarize_messages(
        self,
        messages_to_summarize: List[Union[dict, Any]],
        context_messages: List[Union[dict, Any]] = [],
    ) -> Any:
        """Summarize the messages"""
        from langchain_core.messages import (
            SystemMessage,
            AIMessage,
            ToolMessage,
            HumanMessage,
        )
        import copy

        # Serialize messages into a transcript string with role + tool info
        lines = []
        for msg in messages_to_summarize:
            assert isinstance(msg, ToolMessage), "We only summarize tool messages"
            lines.append(f"[TOOL RESULT]\nname: {msg.name}\ncontent: {msg.content}")
        transcript = "\n\n".join(lines)

        # Steering context: the assistant message(s) that issued the tool call(s)
        # being summarized (e.g. the search query), so the summarizer knows what
        # was being looked for and what to preserve. We intentionally do NOT
        # include the original system prompt or the user's message here.
        tool_call_ids = {
            m.tool_call_id
            for m in messages_to_summarize
            if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None)
        }
        context_source = (
            list(context_messages) if context_messages else list(messages_to_summarize)
        )
        query_lines = []
        seen_ai_ids = set()
        for m in context_source:
            if not isinstance(m, AIMessage) or m.id in seen_ai_ids:
                continue
            matching = [
                tc
                for tc in (getattr(m, "tool_calls", None) or [])
                if tc.get("id") in tool_call_ids
            ]
            if not matching:
                continue
            seen_ai_ids.add(m.id)
            calls_str = "\n".join(
                f"TOOL CALL -> name: {tc.get('name')}, args: {tc.get('args')}"
                for tc in matching
            )
            if any((tc.get("name") == "reflect") for tc in matching):
                # don't compress at all
                return AIMessage(content=transcript)
            query_lines.append(f"{m.content}\n{calls_str}" if m.content else calls_str)
        query_context = (
            "The assistant issued the following tool call(s) that produced the "
            "result below; use it to understand what was being looked for:\n"
            + "\n\n".join(query_lines)
            + "\n\n"
            if query_lines
            else ""
        )

        # Construct exactly two messages
        system_msg = SystemMessage(
            content=_SUMMARY_SYSTEM_PROMPT.format(query_context=query_context)
        )

        user_msg = HumanMessage(
            content=f"Summarize the following tool result:\n\n{transcript}"
        )

        try:
            summary_msg = self._call_summary_model(
                [system_msg, user_msg],
                persist_state=False,
            )
        except Exception as e:
            if not self._is_context_length_error(e):
                raise
            # Transcript too long: truncate each ToolMessage content and retry
            truncated_lines = []
            for i, msg in enumerate(messages_to_summarize):
                if isinstance(msg, ToolMessage):
                    content = (
                        msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
                    )
                    truncated_lines.append(
                        f"[TOOL RESULT]\nname: {msg.name}\ncontent: {content[:1000]}... [truncated]"
                    )
                else:
                    truncated_lines.append(lines[i])
            user_msg = HumanMessage(
                content="Summarize the following conversation:\n\n"
                + "\n\n".join(truncated_lines)
            )
            try:
                summary_msg = self._call_summary_model(
                    [system_msg, user_msg],
                    persist_state=False,
                )
            except Exception as e2:
                if not self._is_context_length_error(e2):
                    raise
                summary_msg = AIMessage(
                    content="[SUMMARY UNAVAILABLE: context too long to summarize]"
                )

        summary_text = (
            f"{_SUMMARY_TAG_PREFIX} {len(messages_to_summarize)} MESSAGES] "
            f"{summary_msg.content}"
        )

        # Single-message input: preserve the original message type and metadata,
        # just replace the content (e.g. keeps tool_call_id on a ToolMessage).
        if len(messages_to_summarize) == 1:
            out = copy.copy(messages_to_summarize[0])
            out.content = summary_text
            return out

        summary_msg.content = summary_text
        return summary_msg

    def _pre_model_compress_hook(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph pre_model_hook: runs before every LLM call in the react loop. When
        self._summarize_state_after is set and at least that many *uncompressed* tool
        results have accumulated, return a shrunk message list for the LLM input only
        (`llm_input_messages`), leaving the persisted state untouched so downstream tool-call
        parsing still sees the real tool responses. No-op (returns {}) when
        disabled, so normal agents are unaffected.
        """
        from langchain_core.messages import ToolMessage

        if self._summarize_state_after is None:
            return {}

        messages = state.get("messages", [])
        n_uncompressed = sum(
            1
            for m in messages
            if isinstance(m, ToolMessage)
            and not str(m.content).lstrip().startswith(_SUMMARY_TAG_PREFIX)
        )
        if n_uncompressed < self._summarize_state_after:
            return {}

        compressed = self._compress_messages(messages)
        return {"llm_input_messages": compressed}

    def _compress_messages(self, messages: List[Any]) -> List[Any]:
        """
        Return a copy of `messages` with each uncompressed ToolMessage replaced by a short
        summary. Pure: does NOT touch the checkpointer/state.

        Tool results are summarized one-for-one, so each summary keeps its original
        `tool_call_id` and every message stays in place — no AI `tool_call` is ever orphaned
        and pairing is preserved by construction. Already-summarized tool messages (tagged)
        and non-tool messages (incl. final AI outputs) are passed through untouched. Summaries
        are cached on self._tool_summary_cache by ToolMessage.id, so a given result is
        summarized at most once across calls.
        """
        import copy as _copy
        from langchain_core.messages import ToolMessage

        def _compress(m: Any) -> Any:
            content = str(m.content).lstrip()
            if not isinstance(m, ToolMessage) or content.startswith(
                _SUMMARY_TAG_PREFIX
            ):
                return m
            if m.id not in self._tool_summary_cache:
                summary = self._summarize_messages([m], context_messages=messages)
                self._tool_summary_cache[m.id] = summary.content
            out = _copy.copy(m)
            out.content = self._tool_summary_cache[m.id]
            return out

        return [_compress(m) for m in messages]

    def compress_state(self):
        """
        Compress the persisted chain by summarizing each tool result (see _compress_messages),
        then write the result back to the checkpointer. Final AI outputs are left untouched.
        """
        if self._pre_compress_hook is not None:
            self._pre_compress_hook()

        og_len = len(self.state)
        if og_len == 0:
            return

        new_state = self._compress_messages(self.state)

        # Reset state
        self.clear_state()
        self.graph.update_state(
            {"configurable": {"thread_id": self.thread_id}},
            {"messages": new_state},
        )

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        """Return True if the exception indicates the context window was exceeded."""
        _CONTEXT_SIGNALS = (
            "context_length_exceeded",
            "prompt is too long",
            "input tokens exceed",
            "too many tokens",
            "maximum context length",
            "context window",
            "reduce the length",
            "input_tokens",
        )
        msg = str(exc).lower()
        return any(sig in msg for sig in _CONTEXT_SIGNALS)

    def _call_graph(
        self,
        messages: List[Union[dict, Any]],
        persist_state: bool = True,
        end_tokens: List[str] = [],  # special tokens which terminate the chain
        num_restarts: int = 0,  # 0: no retry, 1: retry once, 2: retry twice, etc.
        max_react_steps: int = None,
        min_react_steps: int = None,
        tag_on_final_msg: bool = False,
    ):
        """
        Call the graph and return only the new messages generated via this call, EXCLUDING the passed in
                prompts (messages arg), and excluding the previous history up to this point
        This may generate multiple messages until a stop condition is met (by default: no tool calls).

        Args:
            messages: the initial messages (prompt)
            end_tokens: special tokens which terminate the chain
            num_restarts: the number of times to force the chain to continue (until the recursion limit is hit)
            max_react_steps: the maximum number of steps to take (after the initial prompt)
            min_react_steps: the minimum number of steps to take (after the initial prompt)
            tag_on_final_msg: whether to tag the final message
                there should be an automatic tag regardless of this flag in the case of a GraphRecursionError

        Note for chain length:
        We run the chain. When it stops (no tool calls), we check if we should force it to continue:
        - If an end_token is found, we stop the chain.
        - Otherwise, we check if the chain is longer than min_react_steps.
            - If it is, we check if we should force the chain to continue.
                - If max_react_steps is hit, we stop the chain.
                - If num_restarts > 0, we force the chain to continue until max_react_steps is hit.
                - If num_restarts has been exceeded, we stop the chain.
        - If the chain is shorter than min_react_steps, we force the chain to continue.

        Additionally, if tag_on_final_msg=True, we add an extra step to the chain.
        """
        from langgraph.errors import GraphRecursionError
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        # Re-inject system tools message when state is empty (e.g. after clear_state)
        if (
            getattr(self, "_system_tools_msg", None) is not None
            and len(self.state) == 0
        ):
            self.graph.update_state(
                {"configurable": {"thread_id": self.thread_id}},
                {"messages": [SystemMessage(content=self._system_tools_msg)]},
            )

        # Some models (e.g. Anthropic) require at least one human/user message in the
        # full conversation. If the combined state + new messages has no human turn,
        # convert the last system-role entry in `messages` to a user role so the API
        # call doesn't fail with a "first message must be from user" error.
        state_has_human = any(isinstance(m, HumanMessage) for m in self.state)
        new_msgs_have_user = any(
            (isinstance(m, dict) and m.get("role") == "user")
            or isinstance(m, HumanMessage)
            for m in messages
        )
        if not state_has_human and not new_msgs_have_user:
            messages = list(messages)
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, dict) and msg.get("role") == "system":
                    messages[i] = {**msg, "role": "user"}
                    break
                elif isinstance(msg, SystemMessage):
                    messages[i] = HumanMessage(content=msg.content)
                    break

        if max_react_steps is None:
            max_react_steps = self._max_react_steps
        if min_react_steps is None:
            min_react_steps = self._min_react_steps
        assert min_react_steps >= 1, "min_react_steps must be >= 1"
        assert max_react_steps >= min_react_steps, (
            "max_react_steps must be > min_react_steps"
        )

        cfg = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": max_react_steps
            + len(
                messages
            ),  # the recursion_limit considers the entire length of the chain, including the prompt. So if we want to allow the model to take 4 steps after a len-1 prompt, we need to set the recursion limit to 1 + 4 = 5
        }
        og_len = len(self.state)

        def _retry(remaining_retries):
            """Logic for forcing the chain to continue"""
            if any(end_token in self.state[-1].content for end_token in end_tokens):
                return

            num_react_steps = len(self.state) - og_len - len(messages)
            if num_react_steps < min_react_steps:
                # If we're under min_react_steps, force the chain to continue
                continue_chain = True
            elif remaining_retries <= 0:
                # otherwise, if we're over the min number and have exhausted all the retries, finish
                continue_chain = False
            elif cfg["recursion_limit"] - num_react_steps < 1:
                # otherwise, if we're over the min number and have hit the recursion limit, finish
                continue_chain = False
            else:
                # otherwise, we're over the min number and have not hit the recursion limit, continue
                continue_chain = False

            if not continue_chain:
                return

            # force the chain to continue
            cfg["recursion_limit"] = max(1, cfg["recursion_limit"] - num_react_steps)
            continuation_messages = self.state + [
                HumanMessage(
                    content="*SYSTEM NOTE* Use the tools more to improve your response."
                )
            ]
            for attempt in range(2):
                try:
                    _len_before_call = len(self.state)
                    out = self.graph.invoke({"messages": continuation_messages}, cfg)
                    self._raw_state += out["messages"][_len_before_call:]
                    break
                except Exception as exc:
                    if attempt == 0 and self._is_context_length_error(exc):
                        self.compress_state()
                        continuation_messages = self.state + [
                            HumanMessage(
                                content="*SYSTEM NOTE* Use the tools more to improve your response."
                            )
                        ]
                    else:
                        raise
            self._delete_messages(
                ids=self._get_msg_id_by_content(
                    "*SYSTEM NOTE* Use the tools more to improve your response."
                )
            )
            return _retry(remaining_retries - 1)

        def _final_msg(cfg):
            msg = self._call_model_no_tools(
                self.state
                + [
                    HumanMessage(
                        content="*SYSTEM NOTE* In the history above, the user sent a message, and then you had some internal thoughts. You now need to generate a final, user-facing response to the original user message. Note that the user cannot see any of your intermediate thoughts, only this one, so you may need to repeat information above."
                    )
                ],
            )
            self._raw_state += [
                HumanMessage(
                    content="*SYSTEM NOTE* In the history above, the user sent a message, and then you had some internal thoughts. You now need to generate a final, user-facing response to the original user message. Note that the user cannot see any of your intermediate thoughts, only this one, so you may need to repeat information above."
                ),
                msg,
            ]

        ######

        def _invoke_with_context_recovery(invoke_messages):
            """Invoke the graph, recovering once from context-length errors by compressing state."""
            for attempt in range(2):
                try:
                    _len_before_call = len(self.state)
                    out = self.graph.invoke({"messages": invoke_messages}, cfg)
                    self._raw_state += out["messages"][_len_before_call:]
                    return out
                except Exception as exc:
                    if attempt == 0 and self._is_context_length_error(exc):
                        self.compress_state()
                        # The original invoke_messages are already captured in the compressed
                        # state, so retry with no new messages to avoid duplicating them.
                        invoke_messages = []
                    else:
                        raise

        try:
            _invoke_with_context_recovery(messages)
            _retry(num_restarts)
        except GraphRecursionError:
            self._delete_messages(
                ids=self._get_msg_id_by_content(
                    "Sorry, need more steps to process this request."
                )
            )
            _final_msg(cfg)

        if tag_on_final_msg:
            _final_msg(cfg)

        stub = self.state[og_len:]

        # Ensure content is a string
        try:
            ids, contents = [], []
            for m in stub:
                if isinstance(m.content, list):
                    # Preserve messages whose list content carries an Anthropic
                    # cache breakpoint (e.g. the cached system prefix from
                    # fmt_as_dialog) — flattening to a string would strip the
                    # cache_control and break prompt caching on later turns.
                    if any(
                        isinstance(c, dict) and "cache_control" in c
                        for c in m.content
                    ):
                        continue
                    blocks = [
                        c if isinstance(c, dict) else {"type": "text", "text": str(c)}
                        for c in m.content
                    ]
                    texts = [
                        c.get("text", "") for c in blocks if c.get("type") == "text"
                    ]
                    ids.append(m.id)
                    contents.append("\n".join(texts))
            self._update_messages(ids=ids, content=contents)
            stub = self.state[og_len:]
        except Exception as e:
            print(f"Error cleaning message content: {e}")

        if self._out_of_steps_msg is not None:
            _ids = [
                m.id
                for m in stub[:-1]
                if isinstance(m, AIMessage)
                and m.content == "Sorry, need more steps to process this request."
            ]
            self._update_messages(
                ids=_ids,
                content=[self._out_of_steps_msg for _ in _ids],
            )
            stub = self.state[og_len:]

        # Optionally replace empty AI messages with a filler string
        if self._empty_message_filler:
            _ids = [
                m.id
                for m in stub
                if isinstance(m, AIMessage)
                and (m.content is None or str(m.content).strip() == "")
            ]
            if _ids:
                self._update_messages(
                    ids=_ids,
                    content=[self._empty_message_filler for _ in _ids],
                )
                stub = self.state[og_len:]

        if self._add_thinking_tag:
            # This only affects the state, not the returned stub
            self._update_messages(
                ids=[
                    m.id
                    for m in stub[:-1]
                    if isinstance(m, AIMessage)
                    and m.content.strip() != ""
                    and not m.content.startswith(self._thinking_tokens[0])
                ],
                content=[
                    f"{self._thinking_tokens[0]}{m.content}{self._thinking_tokens[1]}"
                    for m in stub[:-1]
                    if isinstance(m, AIMessage)
                    and m.content.strip() != ""
                    and not m.content.startswith(self._thinking_tokens[0])
                ],
            )
            stub = self.state[og_len:]

        # Modify the state (but not the returned stub)
        if not self._multiturn_memory:
            self.clear_state()
            self._raw_state = []

        elif not persist_state:
            self._delete_messages(ids=[m.id for m in stub])

        # Cut out prompt messages before returning
        return stub[len(messages) :]

    def _update_messages(self, ids: List[str], content: List[str]):
        """Change the content of the messages with the given ids"""
        assert len(ids) == len(content), "There should be one content per id"
        new_state = []
        for m in self.state:
            if m.id not in ids:
                new_state.append(m)
            else:
                m.content = content[ids.index(m.id)]
                new_state.append(m)
        self.clear_state()
        self.graph.update_state(
            {"configurable": {"thread_id": self.thread_id}},
            {"messages": new_state},
        )
        return new_state

    def _delete_messages(self, ids: List[str]):
        """Remove messages from the state"""
        from langchain_core.messages import RemoveMessage

        # special edge case: langchain code doesn't handle this well
        # if the ids = the entire state, clear_state() instead
        if ids == [m.id for m in self.state]:
            self.clear_state()
        else:
            self.graph.update_state(
                {"configurable": {"thread_id": self.thread_id}},
                {"messages": [RemoveMessage(id=id) for id in ids]},
            )

    def _get_msg_id_by_content(self, content: str) -> List[str]:
        """Get the ids of the messages with the given content"""
        return [m.id for m in self.state if m.content == content]

    def deduplicate_msgs_by_content(self, contents: set):
        """Delete all but the last occurrence of any message whose content is in `contents`."""
        ids_to_delete = []
        for content in contents:
            matches = [m.id for m in self.state if m.content == content]
            ids_to_delete.extend(matches[:-1])  # keep last, delete all earlier
        if ids_to_delete:
            self._delete_messages(ids_to_delete)

    def insert_message(self, role: str, content: str, persist_state: bool = True):
        """
        Insert a single message into the current conversation state.

        Args:
            role: One of "system", "user", "assistant", or "tool"
            content: Message text content
            persist_state: If True, appends the message to the graph state; if False, no-op to state but returns the constructed message

        Returns:
            The created LangChain BaseMessage
        """
        from langchain_core.messages import (
            SystemMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
        )

        role = role.lower().strip()
        if role == "system":
            msg = SystemMessage(content=content)
        elif role == "user":
            msg = HumanMessage(content=content)
        elif role == "assistant":
            msg = AIMessage(content=content)
        elif role == "tool":
            # Tool messages usually require tool_call_id; we insert bare content if not provided
            msg = ToolMessage(content=content, tool_call_id="manual")
        else:
            raise ValueError(
                "role must be one of 'system', 'user', 'assistant', or 'tool'"
            )

        if persist_state:
            self.graph.update_state(
                {"configurable": {"thread_id": self.thread_id}},
                {"messages": [msg]},
            )
            self._raw_state += [msg]

        return msg


def get_token_usage(
    response_metadata: dict,
    default_value: int = 0,
    return_reasoning_tokens: bool = True,
) -> int:
    """
    Get the token usage from the response metadata.
    If return_reasoning_tokens is True, return completion_tokens + reasoning_tokens.
    Otherwise, return just the completion tokens.
    """
    try:
        # openai
        if return_reasoning_tokens:
            return (
                response_metadata["token_usage"]["completion_tokens"]
                + response_metadata["token_usage"]["completion_tokens_details"][
                    "reasoning_tokens"
                ]
            )
        else:
            return response_metadata["token_usage"]["completion_tokens"]
    except:
        pass

    try:
        # anthropic
        if return_reasoning_tokens:
            return (
                response_metadata["usage"]["output_tokens"]
                + response_metadata["usage"]["output_tokens_details"][
                    "reasoning_tokens"
                ]
            )
        else:
            return response_metadata["usage"]["output_tokens"]
    except:
        pass

    return default_value


def get_token_breakdown(msg) -> Dict[str, int]:
    """Extract per-message token usage broken down into four buckets.

    Buckets:
        - input: total prompt tokens sent to the model (cached + uncached)
        - input_cached: subset of input that was served from prompt cache
        - output: total tokens generated by the model (includes reasoning for OpenAI)
        - reasoning: subset of output tokens used for hidden reasoning (if any)

    Prefers LangChain's normalized ``usage_metadata`` (uniform across OpenAI,
    Anthropic, Gemini). Falls back to provider-raw ``response_metadata`` when
    ``usage_metadata`` is unavailable. Returns zeros if neither is present
    (e.g. local HF/vLLM models that do not report usage).
    """
    breakdown = {"input": 0, "input_cached": 0, "output": 0, "reasoning": 0}

    usage = getattr(msg, "usage_metadata", None) or {}
    if usage:
        breakdown["input"] = int(usage.get("input_tokens", 0) or 0)
        breakdown["output"] = int(usage.get("output_tokens", 0) or 0)
        input_details = usage.get("input_token_details") or {}
        breakdown["input_cached"] = int(input_details.get("cache_read", 0) or 0)
        output_details = usage.get("output_token_details") or {}
        breakdown["reasoning"] = int(output_details.get("reasoning", 0) or 0)
        if any(breakdown.values()):
            return breakdown

    response_metadata = getattr(msg, "response_metadata", None) or {}

    # OpenAI shape
    token_usage = response_metadata.get("token_usage") or {}
    if token_usage:
        breakdown["input"] = int(token_usage.get("prompt_tokens", 0) or 0)
        breakdown["output"] = int(token_usage.get("completion_tokens", 0) or 0)
        prompt_details = token_usage.get("prompt_tokens_details") or {}
        breakdown["input_cached"] = int(prompt_details.get("cached_tokens", 0) or 0)
        completion_details = token_usage.get("completion_tokens_details") or {}
        breakdown["reasoning"] = int(completion_details.get("reasoning_tokens", 0) or 0)
        return breakdown

    # Anthropic shape
    anthropic_usage = response_metadata.get("usage") or {}
    if anthropic_usage:
        input_tokens = int(anthropic_usage.get("input_tokens", 0) or 0)
        cache_read = int(anthropic_usage.get("cache_read_input_tokens", 0) or 0)
        cache_creation = int(anthropic_usage.get("cache_creation_input_tokens", 0) or 0)
        # Anthropic reports the three groups as disjoint; sum to get total input.
        breakdown["input"] = input_tokens + cache_read + cache_creation
        breakdown["input_cached"] = cache_read
        breakdown["output"] = int(anthropic_usage.get("output_tokens", 0) or 0)
        output_details = anthropic_usage.get("output_tokens_details") or {}
        breakdown["reasoning"] = int(output_details.get("reasoning_tokens", 0) or 0)
        return breakdown

    return breakdown


class EmbeddingModelWrapper:
    """
    Wrapper for embedding models to handle different encoding methods.
    Supports standard SentenceTransformer models, Qwen3-Embedding, and EmbeddingGemma models.

    Note: Qwen3-Embedding models require transformers>=4.51.0 and sentence-transformers>=2.7.0.
    EmbeddingGemma models work with standard sentence-transformers installation.

    Includes disk caching for embeddings to avoid recomputing the same texts.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        max_cache_size_mb: Optional[float] = None,
        max_cache_files: Optional[int] = None,
        quantize: Optional[bool] = False,
        batch_size: int = 128,
        cache_queries: bool = False,
        device_map: Optional[str] = None,
        embedding_api_url: Optional[str] = None,
    ):
        """
        Initialize the embedding model wrapper.

        Args:
            model_name: Name of the model (e.g., "all-MiniLM-L6-v2", "Qwen/Qwen3-Embedding-0.6B", "google/embeddinggemma-300m")
            cache_dir: Optional directory for caching embeddings. If None, uses ".cache/embeddings/"
            use_cache: Whether to use disk caching for embeddings (default: True)
            max_cache_size_mb: Maximum cache size in MB. If None, defaults to 1024 MB (1 GB). Set to float('inf') for no limit.
            max_cache_files: Maximum number of cache files. If None, no file count limit (default: None)
            quantize: Quantization level for the model. If None, no quantization is used.
            batch_size: Batch size for encoding. Default is 128 (increased from SentenceTransformer's default of 32)
            cache_queries: Whether to cache query embeddings (default: False)
            device_map: Device map for multi-GPU (e.g. "auto"). Passed to SentenceTransformer via model_kwargs.
            embedding_api_url: If set, use OpenAI-compatible API (e.g. vLLM) instead of loading model locally.
        """
        self.model_name = model_name
        self.embedding_api_url = embedding_api_url

        if embedding_api_url is not None:
            # API mode: use OpenAI client, no local model
            from openai import OpenAI

            base_url = embedding_api_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            self._api_client = OpenAI(
                base_url=base_url,
            )
            self.model = None
        else:
            # Local mode: load SentenceTransformer
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is required. Install it with: pip install sentence-transformers"
                )
            model_kwargs = {}
            if quantize:
                model_kwargs["torch_dtype"] = torch.bfloat16
            if device_map is not None:
                model_kwargs["device_map"] = device_map
            if model_kwargs:
                self.model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
            else:
                self.model = SentenceTransformer(model_name)
            self._api_client = None
        self.use_cache = use_cache
        self.cache_queries = cache_queries
        self.batch_size = batch_size
        self.max_cache_size_mb = (
            float("inf") if max_cache_size_mb is None else max_cache_size_mb
        )
        self.max_cache_files = max_cache_files

        # Set up cache directory
        if cache_dir is None:
            cache_dir = ".cache/embeddings"
        self.cache_dir = Path(cache_dir)
        # Create model-specific subdirectory (sanitize model name for filesystem)
        model_cache_name = model_name.replace("/", "_").replace("\\", "_")
        self.model_cache_dir = self.cache_dir / model_cache_name
        if self.use_cache:
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Detect model type (case-insensitive)
        model_name_lower = model_name.lower()
        self.is_qwen3 = (
            "qwen3-embedding" in model_name_lower
            or "qwen/qwen3-embedding" in model_name_lower
        )
        self.is_embeddinggemma = (
            "embeddinggemma" in model_name_lower
            or "google/embeddinggemma" in model_name_lower
        )

    def _get_cache_path(self, text: str, is_query: bool = False) -> Path:
        """
        Get the cache file path for a given text.

        Args:
            text: The text to embed
            is_query: Whether this is a query (uses encode_query) or document (uses encode)

        Returns:
            Path to the cache file
        """
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        # Include is_query in the filename to distinguish query vs document embeddings
        prefix = "query_" if is_query else "doc_"
        return self.model_cache_dir / f"{prefix}{text_hash}.npy"

    def encode(self, texts, show_progress_bar: bool = False, **kwargs):
        """
        Encode texts into embeddings with caching. Handles different model types appropriately.

        Args:
            texts: List of texts or single text to encode
            show_progress_bar: Whether to show progress bar
            **kwargs: Additional arguments passed to the encoding method

        Returns:
            numpy array of embeddings
        """
        return self._encode_with_cache(
            texts, is_query=False, show_progress_bar=show_progress_bar, **kwargs
        )

    def _encode_with_cache(
        self, texts, is_query: bool = False, show_progress_bar: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Encode texts with caching. Checks cache first, then encodes and saves if not found.

        Args:
            texts: List of texts or single text to encode
            is_query: Whether these are queries (uses encode_query) or documents (uses encode)
            show_progress_bar: Whether to show progress bar during encoding
            **kwargs: Additional arguments passed to the encoding method

        Returns:
            numpy array of embeddings
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        is_single = len(texts) == 1

        if not self.use_cache or (is_query and not self.cache_queries):
            # No caching, just encode directly
            return self._encode_uncached(
                texts, is_query=is_query, show_progress_bar=show_progress_bar, **kwargs
            )

        # Check cache for each text
        embeddings = []
        texts_to_encode = []
        text_indices_to_encode = []

        for i, text in enumerate(texts):
            cache_path = self._get_cache_path(text, is_query=is_query)
            if cache_path.exists():
                # Load from cache
                cached_emb = np.load(cache_path)
                # Ensure it's 2D (num_texts, embedding_dim)
                if cached_emb.ndim == 1:
                    cached_emb = cached_emb.reshape(1, -1)
                embeddings.append(cached_emb)
                # Update access time (mtime) to track LRU
                try:
                    os.utime(cache_path, None)
                except (OSError, FileNotFoundError):
                    # File might have been deleted, continue without updating
                    pass
            else:
                # Need to encode this one
                texts_to_encode.append(text)
                text_indices_to_encode.append(i)
                embeddings.append(None)  # Placeholder

        # Encode texts that weren't in cache
        if texts_to_encode:
            new_embeddings = self._encode_uncached(
                texts_to_encode,
                is_query=is_query,
                show_progress_bar=show_progress_bar,
                **kwargs,
            )

            # Ensure new_embeddings is 2D
            if new_embeddings.ndim == 1:
                new_embeddings = new_embeddings.reshape(1, -1)

            # Save to cache and update embeddings list
            for idx, (text, embedding) in enumerate(
                zip(texts_to_encode, new_embeddings)
            ):
                cache_path = self._get_cache_path(text, is_query=is_query)
                # Save as 1D array for consistency
                np.save(cache_path, embedding.flatten())
                # Update access time (mtime) to track LRU
                os.utime(cache_path, None)
                # Find the original index in the texts list
                orig_idx = text_indices_to_encode[idx]
                embeddings[orig_idx] = embedding.reshape(1, -1)

            # Check cache size and evict if necessary
            self._enforce_cache_limits()

        # Convert to numpy array - stack all embeddings
        if len(embeddings) == 1:
            result = embeddings[0]
        else:
            result = np.vstack(embeddings)

        # Return single embedding if input was single text
        if is_single and result.shape[0] == 1:
            return result[0] if result.ndim == 2 else result
        return result

    def _encode_uncached(
        self, texts, is_query: bool = False, show_progress_bar: bool = False, **kwargs
    ):
        """
        Encode texts without caching. Internal method that does the actual encoding.
        Uses duck typing to check for specialized methods (encode_query, encode_document)
        rather than hardcoding model types.

        Args:
            texts: List of texts to encode
            is_query: Whether these are queries (uses encode_query) or documents (uses encode)
            show_progress_bar: Whether to show progress bar
            **kwargs: Additional arguments passed to the encoding method

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # API mode: use OpenAI-compatible embeddings endpoint
        if self._api_client is not None:
            return self._encode_via_api(
                texts, show_progress_bar=show_progress_bar, **kwargs
            )

        # Local mode: use SentenceTransformer
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self.batch_size

        if is_query:
            if hasattr(self.model, "encode_query"):
                return self.model.encode_query(
                    texts, show_progress_bar=show_progress_bar, **kwargs
                )
            else:
                if self.is_qwen3 and "prompt_name" not in kwargs:
                    kwargs["prompt_name"] = "query"
                return self.model.encode(
                    texts, show_progress_bar=show_progress_bar, **kwargs
                )
        else:
            if hasattr(self.model, "encode_document"):
                return self.model.encode_document(
                    texts, show_progress_bar=show_progress_bar, **kwargs
                )
            else:
                return self.model.encode(
                    texts, show_progress_bar=show_progress_bar, **kwargs
                )

    def _encode_via_api(
        self, texts, show_progress_bar: bool = False, **kwargs
    ) -> np.ndarray:
        """Encode texts via OpenAI-compatible API (e.g. vLLM)."""
        batch_size = kwargs.get("batch_size", self.batch_size)
        all_embeddings = []

        batch_starts = range(0, len(texts), batch_size)
        if show_progress_bar and len(texts) > batch_size:
            try:
                from tqdm import tqdm

                batch_starts = tqdm(batch_starts, desc="Embedding")
            except ImportError:
                pass

        for i in batch_starts:
            batch = texts[i : i + batch_size]
            resp = self._api_client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            for d in resp.data:
                all_embeddings.append(np.array(d.embedding, dtype=np.float32))

        result = np.vstack(all_embeddings)
        if len(texts) == 1:
            return result[0]
        return result

    def encode_query(self, texts, show_progress_bar: bool = False, **kwargs):
        """
        Encode queries into embeddings with caching. Uses model-specific query encoding when available.

        Args:
            texts: List of texts or single text to encode as queries
            show_progress_bar: Whether to show progress bar
            **kwargs: Additional arguments passed to the encoding method

        Returns:
            numpy array of embeddings
        """
        return self._encode_with_cache(
            texts, is_query=True, show_progress_bar=show_progress_bar, **kwargs
        )

    def _enforce_cache_limits(self):
        """
        Enforce cache size limits by evicting least recently used files.
        Uses file modification time (mtime) to determine LRU order.
        """
        if not self.use_cache or not self.model_cache_dir.exists():
            return

        # Get all cache files with their sizes and modification times
        cache_files = []
        total_size_bytes = 0

        for cache_file in self.model_cache_dir.glob("*.npy"):
            try:
                stat = cache_file.stat()
                size_bytes = stat.st_size
                mtime = stat.st_mtime
                cache_files.append((cache_file, size_bytes, mtime))
                total_size_bytes += size_bytes
            except (OSError, FileNotFoundError):
                # File might have been deleted, skip it
                continue

        if not cache_files:
            return

        # Check if we need to evict files
        need_eviction = False

        # Check size limit (skip if max_cache_size_mb is inf)
        if self.max_cache_size_mb != float("inf"):
            max_size_bytes = self.max_cache_size_mb * 1024 * 1024
            if total_size_bytes > max_size_bytes:
                need_eviction = True

        # Check file count limit
        if self.max_cache_files is not None and len(cache_files) > self.max_cache_files:
            need_eviction = True

        if not need_eviction:
            return

        # Sort by modification time (oldest first) for LRU eviction
        cache_files.sort(key=lambda x: x[2])  # Sort by mtime

        # Evict files until we're under the limits
        evicted_size = 0
        evicted_count = 0

        for cache_file, size_bytes, _ in cache_files:
            # Check if we still need to evict
            remaining_size = total_size_bytes - evicted_size
            remaining_count = len(cache_files) - evicted_count

            if self.max_cache_size_mb != float("inf"):
                max_size_bytes = self.max_cache_size_mb * 1024 * 1024
                size_ok = remaining_size <= max_size_bytes
            else:
                size_ok = True  # No size limit
            count_ok = (
                self.max_cache_files is None or remaining_count <= self.max_cache_files
            )

            if size_ok and count_ok:
                break

            # Evict this file
            try:
                cache_file.unlink()
                evicted_size += size_bytes
                evicted_count += 1
            except (OSError, FileNotFoundError):
                # File might have been deleted already, skip it
                pass

        if evicted_count > 0:
            print(
                f"Cache eviction: Removed {evicted_count} files "
                f"({evicted_size / (1024 * 1024):.2f} MB) to stay under limits"
            )

