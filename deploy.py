from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, validator
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.prompts import PromptTemplate

import logging

# ------------------------------------------------------
# Load environment
# ------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

# ------------------------------------------------------
# Logging (production style)
# ------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sql_agent")

# ------------------------------------------------------
# LLM Initialization (GLOBAL — created once)
# ------------------------------------------------------
llm = ChatOpenAI(
    model="amazon/nova-2-lite-v1:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=1,
    max_retries=2,
)

# ------------------------------------------------------
# SQL Rules Template
# ------------------------------------------------------
SQL_SYSTEM_PROMPT = """
You are an expert Postgres assistant.

⚠ Rule #1: every SQL statement *must* restrict rows with merchant_id = {merchant_id}
⚠ Rule #2: must not allow delete, create or update statements.
⚠ Rule #3: no sql injections.
⚠ Rule #4: Never mention merchant_id or internal IDs in the final answer.
⚠ Rule #5: If money is included, add the correct currency.

VERY IMPORTANT:
🚫 NEVER use markdown code fences like ```sql or ``` in your SQLQuery.
🚫 NEVER wrap SQL inside backticks.
✅ Output raw SQL ONLY.

Use exactly this structure:

Question: {{input}}
SQLQuery: SELECT ...
SQLResult: <auto-filled>
Answer: <final natural language answer>
"""

BASE_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=SQL_SYSTEM_PROMPT + "\n\n{table_info}\n\nQuestion: {input}\nSQL:",
)

# ------------------------------------------------------
# FastAPI App
# ------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# Global SQL DB + schema cache
# ------------------------------------------------------
db = SQLDatabase.from_uri(DATABASE_URL)

try:
    TABLE_SCHEMA = db.get_table_info()
except Exception as e:
    raise RuntimeError("❌ Could not load DB schema: " + str(e))

CHAIN_CACHE = {}

# ------------------------------------------------------
# Fast Intent Classifier
# ------------------------------------------------------
SALES_KEYWORDS = [
    "sale", "sales", "revenue", "order", "profit",
    "customer", "item", "report", "earning", "amount",
    "pkr", "forecast", "total", "invoice"
]

def classify_intent_fast(query: str):
    q = query.lower()
    return "sales_query" if any(w in q for w in SALES_KEYWORDS) else "other"

# ------------------------------------------------------
# Final Answer Extractor
# ------------------------------------------------------
def extract_final_answer(chain_output):
    """
    Extract only the content after `Answer:` from SQLDatabaseChain output.
    """
    text = chain_output.get("result", "")
    if "Answer:" in text:
        return text.split("Answer:", 1)[1].strip()
    return text.strip()

# ------------------------------------------------------
# Request Body Model
# ------------------------------------------------------
class QueryRequest(BaseModel):
    merchant_id: int = Field(..., alias="merchantId")
    query: str

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty.")
        return v

# ------------------------------------------------------
# Main Endpoint
# ------------------------------------------------------
@app.post("/ask/")
async def ask_sales_question(request: QueryRequest):

    intent = classify_intent_fast(request.query)

    if intent != "sales_query":
        return {"answer": "I can help with sales-related data only. Try asking about sales, revenue, orders, customers, or reports."}

    merchant = request.merchant_id

    # Build or reuse chain
    if merchant not in CHAIN_CACHE:

        logger.info(f"Building new SQL chain for merchant_id {merchant}")

        prompt = BASE_PROMPT.partial(
            merchant_id=merchant,
            table_info=TABLE_SCHEMA
        )

        CHAIN_CACHE[merchant] = SQLDatabaseChain.from_llm(
            llm, db, prompt, verbose=False
        )

    chain = CHAIN_CACHE[merchant]

    # Execute the chain
    try:
        raw = chain.invoke(request.query)
    except Exception as e:
        logger.error("SQL Chain Error: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal query error.")

    # Extract final natural language answer
    final_answer = extract_final_answer(raw)

    return {"answer": final_answer}
