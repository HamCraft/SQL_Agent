import os
from dotenv import load_dotenv, find_dotenv
from langchain_deepseek import ChatDeepSeek
load_dotenv(find_dotenv())
from langchain_community.utilities import SQLDatabase
from dataclasses import dataclass
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langchain.agents  import create_agent
from langchain_core.messages import SystemMessage
import requests, pathlib
from langgraph.checkpoint.memory import InMemorySaver

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"{local_path} already exists, skipping download.")
else:
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"File downloaded and saved as {local_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

# DB
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

model = ChatDeepSeek(
    model="x-ai/grok-4.1-fast:free",
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
    extra_body={"reasoning": {"enabled": True}},
    temperature=0
)

# Context 
@dataclass
class Context:
    db:SQLDatabase

@tool
def execute_sql(query: str) -> str:
    """ Execute sql command and return results """
    runtime = get_runtime(Context)
    db = runtime.context.db
    try:
        return db.run(query)
    except Exception as e:
        return f"Error {e}"
    

SYSTEM_PROMPT = """You are a careful SQL analyst

Rules:
- Think step-by-step
- WHen you need data, call the tool 'execute_sql' with one SELECT query.
-Read-only, no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
-Limit to 10 rows unless the user explicitly asks otherwise.
-If the tool returns 'Error:' revise the SQL query and try again.
-Preder explicit column lists, avoid general selections SELECT *.

"""    


# Agent
agent = create_agent(
    model=model,
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=Context,
    checkpointer=InMemorySaver()
)

def main():
    user_input = input("Your question: ")
    steps = []

    for step in agent.stream(
        {"messages": [{"role":"user", "content":user_input}]},
        {"configurable": {"thread_id":"0103"}},
        stream_mode= "values",
        context= Context(db=db)
    ):
        step["messages"][-1].pretty_print()
        steps.append(step)


if __name__ == "__main__":
    main()