from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

# Use BaseMessage for more flexibility in the message list
class AgentState(TypedDict):
    messages: List[BaseMessage]

llm = ChatOllama(model="gemma3:1b")

# This function now correctly adds the AI's response to the message history.
def process(state: AgentState) -> AgentState:
    """Invokes the LLM and appends its response to the message history."""
    # The llm.invoke call is correct
    response = llm.invoke(state['messages'])
    
    # CRITICAL FIX: Append the AI's response to the state's messages list
    state['messages'].append(response)
    
    # Return the *modified* state
    return state

# Use a checkpointer to give the agent memory across turns
memory = MemorySaver()

# Build the graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node('process', process)
graph_builder.set_entry_point('process')
graph_builder.add_edge('process', END) # The graph runs once per turn

# Compile the graph with memory
agent = graph_builder.compile(checkpointer=memory)

# 3. IMPLEMENT A CONVERSATIONAL LOOP
# This makes the agent interactive instead of running just once.
thread_config = {"configurable": {"thread_id": "my-thread"}}

print("Agent is ready! Type 'exit' or 'quit' to end the chat.")
while True:
    user_input = input('You: ')
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break
    
    # Invoke the agent. The memory will automatically load and save the messages.
    response = agent.invoke({'messages': [HumanMessage(content=user_input)]}, config=thread_config)
    
    # The final message in the returned state is the AI's response
    ai_response = response['messages'][-1].content
    print(f"AI: {ai_response}")