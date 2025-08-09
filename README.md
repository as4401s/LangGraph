# LangGraph 🦜🔗

**LangGraph** is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It allows you to create agent- and multi-agent workflows with full control and observability.

The key idea is that you can define your application as a graph, where each node is a function or a LangChain component, and the edges define the transitions between them. This allows you to create complex, cyclical data flows that are essential for building sophisticated agentic systems.

---

## Core Concepts

* **Stateful:** LangGraph is designed for building stateful applications. The state is passed between nodes in the graph, and you can use a `Checkpointer` to persist the state and resume the graph from any point.
* **Multi-actor:** LangGraph makes it easy to create multi-agent systems where different agents can collaborate to achieve a goal. Each agent can be a node in the graph, and you can define the transitions between them based on the current state.
* **Cyclical:** Unlike traditional data-flow tools like Airflow or Prefect, LangGraph allows for cyclical data flows. This is essential for building agentic systems where the control flow can be dynamic and unpredictable.
* **Human-in-the-loop:** LangGraph has built-in support for human-in-the-loop workflows. You can use the `HumanInterrupt` class to pause the graph and wait for human input before continuing.

---
