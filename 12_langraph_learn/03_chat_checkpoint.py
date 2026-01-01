# pip install -U pymongo langgraph langgraph-checkpoint-mongodb
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
# define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# define node
def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return { "messages": [response] }

# build graph
graph_builder = StateGraph(State)

# add nodes and edges
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# compile graph
graph = graph_builder.compile()

# function to compile graph with checkpointer
def compile_graph_with_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

# use MongoDB as checkpointer
DB_URI = "mongodb://admin:admin@localhost:27017"

# using the checkpointer in a context manager
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    graph_with_checkpointer = compile_graph_with_checkpointer(checkpointer=checkpointer)

    config = {
            "configurable": {
                "thread_id": "sumit" # user_id
            }
        }


    for chunk in graph_with_checkpointer.stream(
        State({"messages": ["what is my name?"]}),
        config,
        stream_mode="values"
        ):
            chunk["messages"][-1].pretty_print()
    

# (START) -> chatbot -> (END)

# state = { messages: ["Hey there"] }
# node runs: chatbot(state: ["Hey There"]) -> ["Hi, This is a message from ChatBot Node"]

# Checkpointer (piyush) = Hey, My name is Piyush Garg