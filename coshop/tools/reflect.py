"""Chain-of-thought scratchpad tool for shopping-assistant policies.

Provides :func:`get_reflect_tool`, which returns a no-op tool that echoes the
agent's reasoning back to it.  The tool is meant to make the agent's internal
reasoning visible in evaluation traces without affecting the user-facing conversation.
"""


def get_reflect_tool():
    """Return a LangChain tool for chain-of-thought reflection.

    The returned ``reflect`` tool accepts a free-text ``thought`` string and
    echoes it back unchanged.  It is intended as a scratchpad: calling it does
    not affect the user-facing conversation, but the call and its argument are
    recorded in the evaluation trace, making the agent's reasoning inspectable.

    Returns:
        A LangChain ``Tool`` wrapping ``reflect(thought: str) -> str``.
    """
    description = (
        "Reflect on your strategy before taking action (e.g., before searching or making a recommendation). "
        "Reflect both on your current beliefs about the user and available products and your beliefs about how to search for the right products."
    )
    from langchain_core.tools import tool

    @tool(description=description)
    def reflect(thought: str) -> str:
        """Reflect on your strategy before taking action (e.g., before searching or making a recommendation). Reflect both on your current beliefs about the user and available products and your beliefs about how to search for the right products."""
        if thought is None or thought.strip() == "":
            raise ValueError("Thought cannot be None or empty")
        return thought

    return reflect

