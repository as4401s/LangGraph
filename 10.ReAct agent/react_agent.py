# ============================================================
# LangGraph + LangChain (Ollama) toy agent with tool use
# - Adds add/subtract/multiply/divide tools
# - Correctly wires a ToolNode into the graph
# - Uses add_messages to append to state history
# - Clear, precise comments for each step
# ============================================================

from typing import TypedDict, Sequence, Annotated, List

# Base message types shared by Human/AI/System/Tool messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# Decorator to expose Python functions as LLM-callable tools
from langchain_core.tools import tool

# Reducer that tells LangGraph to *append* messages into the state
from langgraph.graph.message import add_messages

# Graph primitives
from langgraph.graph import StateGraph, START, END

# Prebuilt node that runs tools when the LLM requests them
from langgraph.prebuilt import ToolNode

# Ollama chat model wrapper
from langchain_ollama import ChatOllama


# -----------------------------
# 1) State definition
# -----------------------------
class AgentState(TypedDict):
    # Stores a mutable list of messages in the conversation
    # Annotated[..., add_messages] tells LangGraph to append new messages
    messages: Annotated[List[BaseMessage], add_messages]


# -----------------------------
# 2) Tool definitions
# -----------------------------
@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b

@tool
def sub(a: int, b: int) -> int:
    """Subtract b from a and return the difference."""
    return a - b

@tool
def mul(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b

@tool
def div(a: int, b: int) -> float:
    """
    Divide a by b and return the quotient as float.
    Raises ValueError on division by zero to surface a clean tool error.
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


# All tools the model can call
tools = [add, sub, mul, div]


# -----------------------------
# 3) Model binding
# -----------------------------
# Bind the tools to the model so it can emit tool calls that ToolNode understands.
model = ChatOllama(model="qwen3:0.6b").bind_tools(tools)


# -----------------------------
# 4) Agent node: model call
# -----------------------------
def model_call(state: AgentState) -> AgentState:
    """
    Core LLM node:
    - Prepend a SystemMessage to steer behavior.
    - Invoke the model with (system + history).
    - Return ONLY the new AI message; LangGraph will append it to state.
    """
    system_prompt = SystemMessage(
        content="You are my AI assistant. Answer thoroughly and use tools when helpful."
    )
    # state["messages"] already contains the running history; add system at the front
    response = model.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}  # add_messages will merge/append


# -----------------------------
# 5) Control flow: keep going if there are tool calls
# -----------------------------
def should_continue(state: AgentState) -> str:
    """
    Routing function:
    - If the latest AI message contains tool calls, go to the ToolNode.
    - Otherwise, we're done.
    """
    last_message = state["messages"][-1]
    # LangChain chat messages expose `.tool_calls` when the model requests tools
    if getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"


# -----------------------------
# 6) Graph wiring
# -----------------------------
graph = StateGraph(AgentState)

# Register our agent (LLM) node
graph.add_node("agent", model_call)

# Register a ToolNode that knows how to execute our Python tools
graph.add_node("tools", ToolNode(tools))

# Start -> agent
graph.add_edge(START, "agent")

# Agent -> (conditional) either tools or END
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",  # if tool calls present, run them
        "end": END,           # otherwise, finish
    },
)

# Tools -> agent (after executing tools, hand results back to model)
graph.add_edge("tools", "agent")

# Compile the graph into an app
app = graph.compile()


# -----------------------------
# 7) Streaming helper (pretty-print)
# -----------------------------
def print_stream(stream):
    """
    Utility to iterate over streaming values from the graph and print the
    newest message each time.
    """
    for step_state in stream:
        msg = step_state["messages"][-1]
        # pretty_print() shows role and content nicely in the console
        msg.pretty_print()


# -----------------------------
# 8) Example run
# -----------------------------
if __name__ == "__main__":
    # Prefer explicit HumanMessage in state (tuples are accepted in many APIs,
    # but BaseMessage objects are safest for graph state).
    user_prompt = (
        "Add 99 + 66, then multiply the result by 2, "
        "then divide by 3. Also tell me a joke, please."
    )
    inputs = {"messages": [HumanMessage(content=user_prompt)]}

    # Stream the run so you can see the tool call(s) and final response
    print_stream(app.stream(inputs, stream_mode="values"))


"""
================================ Human Message =================================

Add 99 + 66, then multiply the result by 2, then divide by 3. Also tell me a joke, please.
================================== Ai Message ==================================

<think>
Okay, let's break down the user's request. They want me to do three things: first add 99 and 66, then multiply the result by 2, then divide by 3. Also, they want me to tell a joke. 

First, I need to handle the math operations step by step. Adding 99 and 66 would be straightforward. Let me calculate that. 99 + 66 equals 165. Then multiply that by 2, which gives 330. Finally, divide 330 by 3, which is 110. So those three operations should be done in order. Once I get the results, I should present them in a clear way. 

But wait, the user also wants a joke. Hmm, I need to come up with a joke that's appropriate. Maybe something like "What do you call a dog that plays soccer? A soccer dog!" But I should make sure it's a fun and nonsensical joke. Once all the math is done, I can present the final answer and the joke.
</think>
Tool Calls:
  add (4ffcc20f-f9ce-4e9b-84c5-de411e67900c)
 Call ID: 4ffcc20f-f9ce-4e9b-84c5-de411e67900c
  Args:
    a: 99
    b: 66
  mul (ae1e541d-1d69-4341-9af6-5e7f40080a30)
 Call ID: ae1e541d-1d69-4341-9af6-5e7f40080a30
  Args:
    a: 165
    b: 2
  div (d8a5727d-7781-442e-bc56-38fd139fdcc9)
 Call ID: d8a5727d-7781-442e-bc56-38fd139fdcc9
  Args:
    a: 330
    b: 3
================================= Tool Message =================================
Name: div

110.0
================================== Ai Message ==================================

<think>
Okay, let me check the calculations again to be sure. First, adding 99 and 66 gives 165. Then multiplying by 2 is 330. Dividing by 3 is 110. All steps seem correct. Now, for the joke: "What do you call a dog that plays soccer? A soccer dog!" That's a play on words, so it's a good joke. I need to present all results and the joke clearly. No errors in the math, so I'll go with that.
"""

