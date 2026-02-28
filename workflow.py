import functools
import logging

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import initialize_model
from prompts import SYSTEM_PROMPT
from tools import fill_field, type_and_select, select_option, fill_ui_select, add_diagnosis, fill_dx_pointers, done_filling

logger = logging.getLogger(__name__)

AGENT_TOOLS = [fill_field, type_and_select, select_option, fill_ui_select, add_diagnosis, fill_dx_pointers, done_filling]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@functools.cache
def create_automation_agent():
    """Build and compile the LangGraph agent. Cached — model is initialised only once."""
    llm = initialize_model()
    tool_node = ToolNode(AGENT_TOOLS)

    async def call_model(state: AgentState) -> dict:
        all_msgs = list(state["messages"])
        if len(all_msgs) > 60:
            all_msgs = all_msgs[:1] + all_msgs[-58:]

        response = await llm.bind_tools(AGENT_TOOLS, parallel_tool_calls=False).ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + all_msgs
        )

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tc = response.tool_calls[0]
            args_str = ', '.join(
                f"{k}={repr(v[:60]) if isinstance(v, str) else v}"
                for k, v in tc.get('args', {}).items()
            )
            logger.info("  Tool call: %s(%s)", tc.get('name'), args_str)

        return {"messages": [response]}

    async def call_tools(state: AgentState) -> dict:
        result = await tool_node.ainvoke(state)
        messages_to_add = list(result.get("messages", []))

        for msg in messages_to_add:
            logger.info("  → %s", (msg.content if hasattr(msg, 'content') else str(msg))[:150])

        if messages_to_add:
            last_content = getattr(messages_to_add[-1], 'content', '') or ''
            if last_content.startswith("Failed") or last_content.startswith("No "):
                failed_tool = next(
                    (m.tool_calls[-1].get('name', '') for m in reversed(state["messages"])
                     if hasattr(m, 'tool_calls') and m.tool_calls),
                    ""
                )
                messages_to_add.append(HumanMessage(content=(
                    f"'{failed_tool}' just failed: \"{last_content[:200]}\". "
                    "Do NOT call done_filling yet. Retry that same tool once with "
                    "corrected arguments (e.g. a different selector, shorter search "
                    "text, or adjusted value) to fix the issue. "
                    "Only move to the next field if it fails a second time."
                )))

        return {"messages": messages_to_add}

    def route_after_agent(state: AgentState) -> str:
        for msg in reversed(state["messages"]):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                return "tools"
            if hasattr(msg, 'content'):
                return END
        return END

    def route_after_tools(state: AgentState) -> str:
        for msg in reversed(state["messages"]):
            if hasattr(msg, 'content') and msg.content == "DONE":
                return END
            if hasattr(msg, 'tool_calls'):
                break
        return "agent"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_after_agent)
    workflow.add_conditional_edges("tools", route_after_tools)
    return workflow.compile()


async def run_phase(graph, messages: list, phase_name: str, recursion_limit: int = 50) -> list:
    logger.info("%s", phase_name)
    logger.info("%s", "─" * 60)
    collected: list = []
    try:
        async for chunk in graph.astream(
            {"messages": messages},
            {"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            if "messages" in chunk:
                collected = chunk["messages"]
        return collected
    except Exception:
        if collected:
            return collected
        raise
