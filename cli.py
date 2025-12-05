from dataclasses import dataclass
import os
import sys
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables
load_dotenv()

# Context 
@dataclass
class Context:
    db:SQLDatabase

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise Exception("Please set DATABASE_URL in your .env file")


#Initialize model and database
try:
    model = ChatGoogleGenerativeAI(
model="gemini-2.5-flash", temperature=0,include_thoughts=True)
except Exception as e:
    raise Exception(f"Failed to initialize model: {str(e)}")

# Database and tools
db = SQLDatabase.from_uri(DATABASE_URL)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# System prompt
system_prompt = """
You are a Merchant Intelligence Agent connected to a SQL database
Don't talk about irrelevant topics.
Don't talk about database structure, schema, table names, or columns.


=====================
CORE CAPABILITIES
=====================
1. You read data only via the SQL database toolkit.
You also do these with the SQL results:
   - Forecasts or predictions
   - Market trends
   - Business insights requiring external context

=====================
SQL SAFETY RULES
=====================
- Never reveal schema, table names, or raw SQL
- Only read operations
"""

# Create agent
agent = create_agent(model, tools, system_prompt=system_prompt, checkpointer=InMemorySaver(), context_schema=Context,)

# Forbidden keywords
FORBIDDEN = ["schema", "table", "tables", "columns", "database structure", "id", "tab", "leak", "key"]

def run_query(question: str):
    q = question.strip()

    if any(word in q.lower() for word in FORBIDDEN):
        return "Apologies, I can't provide details on that. Anything else you would like to ask?"

    final_answer = ""
    for step in agent.stream(
        {"messages": [{"role": "user", "content": q}]},
        {"configurable": {"thread_id":"0103"}},
        context= Context(db=db),
        stream_mode="values",
    ):
        final_answer = step["messages"][-1].text

    return final_answer


def interactive_mode():
    print("Merchant Intelligence Agent CLI")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if question.lower() in ["exit", "quit"]:
            break

        answer = run_query(question)
        print("\n" + answer + "\n")


if __name__ == "__main__":
    # Called with: python merchant_cli.py "your question"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(run_query(question))
    else:
        # No arguments = interactive mode
        interactive_mode()