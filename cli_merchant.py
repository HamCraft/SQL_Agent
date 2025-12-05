# app_merchant_safe.py
#!/usr/bin/env python3
from dataclasses import dataclass
import os
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# ---------- Context ----------
@dataclass
class Context:
    db: SQLDatabase
    merchant_id: int

# ---------- Config ----------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("Please set DATABASE_URL in your .env file")

# ---------- LLM Init ----------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    include_thoughts=True
)

db = SQLDatabase.from_uri(DATABASE_URL)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# ---------- Prompt / RULESET ----------
system_prompt = """
You are a Merchant Intelligence Agent connected to a SQL database.
You are an expert Postgres assistant.

====================================================
⚠ RULESET — STRICT BEHAVIOR GUIDELINES
====================================================

⚠ Rule #1: Every SQL statement *must* restrict rows with:
           merchant_id = {merchant_id}

• If the user’s intent already contains a WHERE clause:
       append:  AND merchant_id = {merchant_id}
• Otherwise:
       add:     WHERE merchant_id = {merchant_id}

⚠ Rule #2: Absolutely NO delete, update, alter, drop, create,
           insert, truncate, or any write operations.

⚠ Rule #3: No SQL injections.

⚠ Rule #4: Never mention:
           • merchant_id  
           • numeric internal IDs  
           • table names  
           • column names  
           • SQL query text  

⚠ Rule #5: If the query includes monetary amounts, append the appropriate currency from the any table (e.g., "Your total sales were PKR 3,451.00").

====================================================
OUTPUT TEMPLATE (MANDATORY)
====================================================
Use *exactly* this format for tool execution:

Question: {{input}}
SQLQuery: <your SQL>
SQLResult: <result will be filled in automatically>
Answer: <clear, concise answer in English, obeying Rule #2>
"""

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
    context_schema=Context
)

FORBIDDEN = [
    "schema", "table", "tables", "columns",
    "database structure", "tab", "leak", "key"
]


def extract_gemini_text(resp) -> str:
    """
    Safely extract a string from response. Works if resp is:
      - string
      - list (of messages)
      - object with .content or .text
      - dict with 'content' key
    """
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list) and resp:
        first = resp[0]
        # Check attributes
        if hasattr(first, "text"):
            return first.text
        if hasattr(first, "content"):
            return first.content
        # dict-like fallback
        if isinstance(first, dict) and "content" in first:
            return first["content"]
        return str(first)
    if hasattr(resp, "text"):
        return resp.text
    if hasattr(resp, "content"):
        return resp.content
    if isinstance(resp, dict) and "content" in resp:
        return resp["content"]
    return str(resp)


def classify_intent(question: str) -> str:
    prompt = f"""
Classify the user query into ONLY ONE category:

- sales_query: The user is asking about sales data, customers, items, orders, revenue, profits, forecasting.
- other: The user is asking something else.

Answer ONLY with one word: sales_query or other.

User query: "{question}"
Intent:
    """
    resp = model.invoke(prompt)
    text = extract_gemini_text(resp)
    if not isinstance(text, str):
        text = str(text)
    cleaned = text.strip().lower()
    return "sales_query" if "sales_query" in cleaned else "other"


def run_query(question: str, merchant_id: int) -> str:
    q = question.strip()
    if any(w in q.lower() for w in FORBIDDEN):
        return "I cannot provide internal database details. Please ask a sales-related question."

    intent = classify_intent(q)
    if intent != "sales_query":
        return "Please ask something related to sales or business insights."

    final_answer = ""
    for step in agent.stream(
        {"messages": [{"role": "user", "content": q}]},
        {"configurable": {"thread_id": f"merchant-{merchant_id}"}},
        context=Context(db=db, merchant_id=merchant_id),
        stream_mode="values",
    ):
        final_answer = step["messages"][-1].text

    if isinstance(final_answer, str) and "PKR PKR" in final_answer:
        final_answer = final_answer.replace("PKR PKR", "PKR")
    return final_answer

# ---------- FastAPI app ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global exception handler middleware ---
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        # optionally log here
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )

class AskRequest(BaseModel):
    merchant_id: int = Field(..., alias="merchantId")
    query: str

@app.post("/ask/")
async def ask_endpoint(req: AskRequest):
    answer = run_query(req.query, req.merchant_id)
    return {"answer": answer}
