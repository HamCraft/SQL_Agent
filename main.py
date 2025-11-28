# import pathlib
# import requests
# from dotenv import load_dotenv
# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents import create_agent

# # Load environment variables
# load_dotenv()

# # Download database if it doesn't exist
# url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
# local_path = pathlib.Path("Chinook.db")
# if not local_path.exists():
#     response = requests.get(url)
#     if response.status_code == 200:
#         local_path.write_bytes(response.content)
#         print(f"Database downloaded: {local_path}")
#     else:
#         raise Exception(f"Failed to download database (status {response.status_code})")

# # Initialize model and database
# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
# db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# toolkit = SQLDatabaseToolkit(db=db, llm=model)
# tools = toolkit.get_tools()

# # Updated system prompt to prevent exposing DB internals
# system_prompt = f"""

# You are a merchant agent designed to answer questions about the data in a SQL database.

# Do NOT reveal table names, column names, database schema, or any internal database details.

# You can only provide answers to user questions using the data content.

# Always limit query results to at most 5 rows.

# Do NOT run any DML statements (INSERT, UPDATE, DELETE, DROP, etc.).

# If a question asks for internal structure, respond politely that you cannot reveal it.

# """

# # Create agent
# agent = create_agent(model, tools, system_prompt=system_prompt)

# # FastAPI app
# app = FastAPI(title="SQL Agent API")

# # Enable CORS if needed
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Request model
# class QueryRequest(BaseModel):
#     question: str

# # API endpoint
# @app.post("/ask")
# async def ask_question(request: QueryRequest):
#     question = request.question.strip()
    
#     # Simple sanitization: block questions explicitly asking for schema or table names
#     forbidden_keywords = ["schema", "table", "tables", "columns", "database structure"]
#     if any(word in question.lower() for word in forbidden_keywords):
#         return {"answer": "I'm sorry, I cannot provide internal database structure details."}

#     final_answer = ""
#     for step in agent.stream(
#         {"messages": [{"role": "user", "content": question}]},
#         stream_mode="values",
#     ):
#         final_answer = step["messages"][-1].text
#     return {"answer": final_answer}


from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent

# Load environment variables
load_dotenv()

# Read full DATABASE_URL from .env
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("Please set DATABASE_URL in your .env file")

# Initialize model and database
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    )


db = SQLDatabase.from_uri(DATABASE_URL)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# System prompt to prevent exposing DB internals
system_prompt = f"""
Answer questions about the data only.
Do NOT reveal table names, column names, or schema.
Limit query results to 5 rows max.
Do NOT run INSERT, UPDATE, DELETE, DROP, or other DML.
If asked about structure, politely refuse.
Answer concisely.
DO Not talk about irrelevant stuff.
"""

# Create agent
agent = create_agent(
    model, 
    tools, 
    system_prompt=system_prompt)

# FastAPI app
app = FastAPI(title="Postgres SQL Agent API")

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    question: str

# API endpoint
@app.post("/ask")
async def ask_question(request: QueryRequest):
    question = request.question.strip()
    
    # Simple sanitization: block questions explicitly asking for schema or table names
    forbidden_keywords = ["schema", "table", "tables", "columns", "database structure"]
    if any(word in question.lower() for word in forbidden_keywords):
        return {"answer": "Apologies, I can't provide details on that. Is there anything else I can assist you with or any other questions you have?"}

    final_answer = ""
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        final_answer = step["messages"][-1].text
    return {"answer": final_answer}

# This is important for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
