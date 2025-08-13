# Step 1: Import necessary libraries
# ------------------------------------
import json
import os
from typing import List, Union, TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage, messages_to_dict, messages_from_dict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama

# --- NEW: Define a constant for our history file ---
HISTORY_FILE = "conversation_history.json"

# Step 2: Define the Agent's State (updated to use add_messages)
# --------------------------------------------------------------
class AgentState(TypedDict):
    """Represents the state of our agent."""
    messages: Annotated[List[Union[HumanMessage, AIMessage]], add_messages]

# --- Helper functions for saving and loading history ---

def save_history(messages: List[Union[HumanMessage, AIMessage]]):
    """
    Saves the conversation history to a JSON file.
    """
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        dict_messages = messages_to_dict(messages)
        json.dump(dict_messages, f, indent=4, ensure_ascii=False)

def load_history() -> List[Union[HumanMessage, AIMessage]]:
    """
    Loads the conversation history from a JSON file.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            dict_messages = json.load(f)
        # Convert dictionaries back into Message objects
        return messages_from_dict(dict_messages)  # <-- correct counterpart
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        # Handle empty/corrupted files gracefully
        return []

# Step 3: Initialize the Language Model
# ---------------------------------------
llm = ChatOllama(model="gemma3:1b")

# Step 4: Define the Graph's Nodes
# --------------------------------
def process(state: AgentState) -> AgentState:
    """Invoke the LLM with the current conversation history and append the reply."""
    response = llm.invoke(state["messages"])
    ai_msg = AIMessage(content=response.content)
    # Because we declared messages with add_messages, returning a state with
    # the new message will merge it into the history.
    return {"messages": [ai_msg]}

# Step 5: Build and Compile the Graph
# -----------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("process", process)
workflow.add_edge(START, "process")
workflow.add_edge("process", END)
agent = workflow.compile()

# Step 6: Create the Conversation Loop
# ------------------------------------
conversation_history = load_history()

if conversation_history:
    print(f"✅ Loaded {len(conversation_history)} messages from previous sessions.")
else:
    print("✨ Starting a new conversation.")

print("Type 'exit' to end the conversation and save the history.")

while True:
    user_input = input("You: ")
    if user_input.lower().strip() == "exit":
        print("\n💾 Saving conversation history...")
        save_history(conversation_history)
        print("Exiting conversation. Goodbye! 👋")
        break

    # Append user message to history
    conversation_history.append(HumanMessage(content=user_input))

    # Run the graph with the current history
    result = agent.invoke({"messages": conversation_history})

    # The compiled graph returns the merged state; extract messages
    conversation_history = result["messages"]

    # Print the latest AI message (the one we just appended in process)
    print(f"\nAI: {conversation_history[-1].content}")


"""
✨ Starting a new conversation.
Type 'exit' to end the conversation and save the history.
You: hey. how are you doing today

AI: Hey there! I’m doing pretty well, thanks for asking! As an AI, I don’t really *feel* things the way humans do, but I’m running smoothly and ready to help out with whatever you need. 😊 

How about you? How’s your day going so far?
You: i am arjun and i am having a great day

AI: That's fantastic to hear, Arjun! It’s wonderful to hear you’re having a great day. That’s really great! 😄  Is there anything you’re looking forward to today?
You: exit

💾 Saving conversation history...
Exiting conversation. Goodbye! 👋

"""