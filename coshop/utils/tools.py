"""
Adapted from https://github.com/volcengine/verl/blob/main/verl/experimental/agent_loop/tool_parser.py
"""

from langchain_core.messages import BaseMessage
import re
import json
import uuid
from langchain_core.messages import AIMessage, RemoveMessage
from langchain_core.messages.tool import InvalidToolCall, ToolCall

HERMES_PROMPT = """You have access to the following tools:\n{tools}\n\nOnly choose from the following tools: {tool_names}."""


def extract_tool_calls(response: str) -> tuple[str, list[str]]:
    """
    Hermes tool parsing.
    """
    tool_call_regex = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    matches = tool_call_regex.findall(response)
    function_calls = []
    for match in matches:
        function_calls.append(match)
    content = tool_call_regex.sub("", response)
    return content, function_calls


def post_model_parse_tools_hook(state: dict) -> dict:
    """
    Hermes tool parsing hook.
    """
    content, tool_calls_str = extract_tool_calls(state["messages"][-1].content)

    # build new message
    tool_calls, invalid_tool_calls = [], []
    for tool_call_str in tool_calls_str:
        error = None
        try:
            js = json.loads(tool_call_str)
            name = js["name"]
            args = js["arguments"]
        except json.JSONDecodeError as e:
            error = f"Invalid JSON: {e}"
            name = tool_call_str
            args = "{}"

        if error:
            invalid_tool_calls.append(
                InvalidToolCall(
                    name=name,
                    args=args,
                    id=str(uuid.uuid4()),
                    error=error,
                )
            )
        else:
            tool_calls.append(
                ToolCall(
                    name=name,
                    args=args,
                    id=str(uuid.uuid4()),
                )
            )

    message = AIMessage(
        content=content,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )

    # replace last message with new message
    last_id = state["messages"][-1].id
    update = {"messages": [RemoveMessage(id=last_id), message]}
    return update


from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


class ChatHuggingFaceTools(ChatHuggingFace):
    def _to_chat_prompt(
        self,
        messages: list[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            msg = "At least one HumanMessage must be provided!"
            raise ValueError(msg)

        if not isinstance(messages[-1], (HumanMessage, ToolMessage)):
            msg = "Last message must be a HumanMessage or ToolMessage!"
            raise ValueError(msg)

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            msg = f"Unknown message type: {type(message)}"
            raise ValueError(msg)

        return {"role": role, "content": message.content}
