# LangGraph 🦜🔗

**LangGraph** is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It allows you to create agent- and multi-agent workflows with full control and observability.

The key idea is that you can define your application as a graph, where each node is a function or a LangChain component, and the edges define the transitions between them. This allows you to create complex, cyclical data flows that are essential for building sophisticated agentic systems.

---

# LangGraph Examples — Guided README

This repo is a hands-on playground for learning **LangGraph** (graphs for LLM workflows on top of LangChain) using small, focused notebooks and a few Python agents. The examples build up from “hello world” graphs to conditional routing, loops, and a ReAct-style tool-using agent.

On GitHub you’ll see top-level notebooks like `1.Hello_world_langgraph.ipynb` through `7.Looping_graph.ipynb`, plus folders for larger examples such as `8.AI_agent_1`, `9.Chatbot`, and `10.ReAct agent`.

> The root README currently describes LangGraph at a high level; this document goes deeper into what each file/folder here actually does.

---

## Quick Start

### Requirements (suggested)
- Python 3.10+
- `pip install langchain langgraph langchain-ollama pydantic charset-normalizer`  
  > `charset-normalizer` (or `chardet`) silences a common `RequestsDependencyWarning`.
- **Ollama** (for local models): <https://ollama.com/>  
  - Pull a tool-capable model, e.g.:
    ```bash
    ollama pull qwen2.5
    # or
    ollama pull llama3.1:8b-instruct
    ```
  - Some very small models (e.g., `gemma3:1b`) **don’t support tools** and will error if you bind tools to them.

### Running
- Open the `.ipynb` notebooks in Jupyter / VS Code and run cells top-to-bottom.
- For Python examples inside folders (e.g., `10.ReAct agent/react_agent.py`), run:
  ```bash
  python react_agent.py
  ```

---

## Repository Map & What Each Example Teaches

### 1. `1.Hello_world_langgraph.ipynb`
**Goal:** Smallest possible LangGraph.  
**What it likely shows:**  
- Define a `TypedDict` state with a `messages` list.
- Create a single node (a Python function) that returns a response.
- Wire edges: `START -> node -> END`.
- Run once to see a basic request/response.  
**Concepts:** State passing, node functions, compiling & invoking a graph.

---

### 2. `2.Multi_input_graph.ipynb`
**Goal:** Feed multiple inputs into nodes or combine state from different sources.  
**What it likely shows:**  
- Multiple fields in the state (or multiple upstream nodes contributing to a single node).
- How LangGraph merges state between nodes.  
**Concepts:** Multi-input shaping, state merging.

---

### 3. `3.Graph_with_condition.ipynb`
**Goal:** Route the flow based on runtime state.  
**What it likely shows:**  
- A router function that inspects state and returns a label (e.g., `"branch_a"` or `"branch_b"`).
- `add_conditional_edges` to send the graph down different paths.  
**Concepts:** Conditional routing, branching control flow.

---

### 4. `4.Sequential_Graph.ipynb`
**Goal:** Chain multiple steps in order.  
**What it likely shows:**  
- Several nodes where each performs a small part (e.g., parse → compute → format).
- Edges connecting them sequentially.  
**Concepts:** Linear pipelines, step decomposition.

---

### 5. `5.Conditional_nodes_in_graph.ipynb`
**Goal:** Add optional steps depending on state.  
**What it likely shows:**  
- Conditions guarding specific nodes (e.g., “only run this validator if X is present”).  
**Concepts:** Optional execution, guarded transitions.

---

### 6. `6.Cinditional_nodes_extended.ipynb`
**Goal:** Extend conditional logic (spelled “Cinditional” in filename).  
**What it likely shows:**  
- More involved routing: multiple branches, nested conditions, or fallbacks.  
**Concepts:** Complex routing patterns.

---

### 7. `7.Looping_graph.ipynb`
**Goal:** Demonstrate cycles/loops in LangGraph.  
**What it likely shows:**  
- A node that re-routes back to itself or a previous node until a termination criterion is met (e.g., “keep refining until score > threshold”).  
**Concepts:** Cyclical graphs, termination checks.

---

### 8. `8.AI_agent_1/`
**Goal:** A first “agent” that uses an LLM to decide next actions.  
**What it likely contains:**  
- A minimal state (messages), a model call, and basic control flow.
- May demonstrate **checkpointers** or **interrupts** for human-in-the-loop.  
**Concepts:** Agent loop structure, state + model invocation.

---

### 9. `9.Chatbot/`
**Goal:** Simple chatbot with conversation memory.  
**What it likely contains:**  
- A state that accumulates `messages` and a single model node.
- Printing/streaming helpers for pretty console output.  
**Concepts:** Persistent message history, minimal chat agent.

---

### 10. `10.ReAct agent/`
**Goal:** ReAct-style agent that calls **tools** (functions) via the LLM.  
**What it contains / what you’ll run:**  
- A Python script like `react_agent.py` that:
  - Defines tools such as `add`, `sub`, `mul`, `div` (decorated with `@tool`).
  - Binds tools to an Ollama chat model: `ChatOllama(...).bind_tools(tools)`.
  - Wires a **ToolNode** so that when the model emits `tool_calls`, the graph executes them and feeds results back to the model.
  - Uses `Annotated[List[BaseMessage], add_messages]` for the `messages` state so each node can return only the **new** message and LangGraph appends it.  
**Concepts:** Tool use, conditional routing (`should_continue`), agent loops, integration with Ollama.

---

## Key Patterns Used in This Repo

- **State as a TypedDict**  
  Define a schema for what flows between nodes. The most common field here is `messages`.

- **`add_messages` reducer**  
  Declare your state field with:  
  ```python
  from typing import Annotated, List
  from langchain_core.messages import BaseMessage
  from langgraph.graph.message import add_messages

  class AgentState(TypedDict):
      messages: Annotated[List[BaseMessage], add_messages]
  ```
  This tells LangGraph to **append** new messages returned by nodes, instead of overwriting the list.

- **Model node that returns only the new message**  
  ```python
  def model_call(state: AgentState) -> AgentState:
      response = model.invoke(state["messages"])
      return {"messages": [response]}
  ```

- **Conditional routing**  
  Use a function (e.g., `should_continue`) to inspect the last message and route:
  ```python
  graph.add_conditional_edges(
      "agent",
      should_continue,
      {"continue": "tools", "end": END},
  )
  ```

- **Tool execution with `ToolNode`**  
  Create a `ToolNode(tools)` and add an edge `tools -> agent` so results flow back for the final answer.

---

## Common Pitfalls & How to Avoid Them

- **Model doesn’t support tools**  
  If you bind tools to a model that lacks tool/function-calling in Ollama (e.g., `gemma3:1b`), you’ll see an error like:  
  *“does not support tools (status code: 400)”*  
  **Fix:** use a tool-capable model such as `qwen2.5` or `llama3.1:8b-instruct`.

- **Requests warning about character detection**  
  If you see:
  ```
  RequestsDependencyWarning: Unable to find acceptable character detection dependency (chardet or charset_normalizer).
  ```
  install one of them:
  ```bash
  pip install charset-normalizer
  # or
  pip install chardet
  ```

- **Using `Sequence` instead of `List` in state**  
  Prefer `List[BaseMessage]` for `messages`. `Sequence` is read-only and can confuse IDE/type checkers in a mutable graph.

---

## Tips for Extending

- Add more tools (e.g., web search, file I/O) and extend your `should_continue` logic to enable multi-step tool plans.
- Introduce a **score** in state and a loop that keeps refining until `score >= threshold` (see `7.Looping_graph.ipynb` for looping patterns).
- Capture **checkpoints** to resume long agent runs or allow human approval steps.

---

## Repo At-a-Glance

- Example notebooks: `1` through `7` including conditional and looping examples.
- Example folders: `8.AI_agent_1`, `9.Chatbot`, `10.ReAct agent`.
- Languages: ~87% Jupyter Notebook, ~13% Python at the time of writing.

---
