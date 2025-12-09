from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, validator
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # load .env

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
MERCHANT_ID = int(os.getenv("MERCHANT_ID", 1))

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Please define it in your .env file.")

SQL_SYSTEM_PROMPT = """
You are an expert Postgres assistant.

⚠  Rule #1: every SQL statement *must* restrict rows with
        merchant_id = {merchant_id}
• If the user’s intent already creates a WHERE clause, append
  AND merchant_id = {merchant_id}.
• Otherwise add WHERE merchant_id = {merchant_id}.

⚠  Rule #2: must not allow delete, create or update statements.

⚠  Rule #3: no sql injections.

⚠  Rule #4 **Never mention merchant_id, numeric IDs, or internal field names**
            in the final Answer. Present a clean, business-friendly reply
            (e.g. “Your total sales were $3,451.00”).  

⚠  Rule #5: If the query includes monetary amounts, append the appropriate currency from the table (e.g. “Your total sales were PKR 3,451.00”).

Use *exactly* this output template so the system can run the query:

Question: {{input}}
SQLQuery: <your SQL>
SQLResult: <result will be filled in automatically>
Answer: <clear, concise answer of the SQLResult in English, give it in human words obeying Rule #2>
"""

BASE_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=SQL_SYSTEM_PROMPT + "\n\n{table_info}\n\nQuestion: {input}\nSQL:",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify domains "*,*" "*","*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    merchant_id: int = Field(..., alias="merchantId")
    query: str

    @validator("query")
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v

def classify_intent(llm: ChatGoogleGenerativeAI, query: str) -> str:
    # Build message list following LLM API
    messages = [
        ("system", "You are an assistant that classifies user queries into two categories ONLY:\n\n"
            "- sales_query: The user is asking about sales data, customer,phone number items, report, revenue, orders, profits, sales formula, forecasting formula or anything related to sales reports or customer or items or forecasting of sales.\n"
            "- other: The user is asking about something else.\n\n"
            "Answer ONLY with one word: sales_query or other."),
        ("human", f'User query: "{query}"\nIntent:')
    ]
    ai_resp = llm.invoke(messages)
    # For Gemini-based LLMs, use .text to get the answer string
    intent = ai_resp.text.strip().lower()
    if intent not in {"sales_query", "other"}:
        intent = "other"
    return intent

@app.post("/ask/")
async def ask_sales_question(request: QueryRequest):
    # Connect to the database

    db = SQLDatabase.from_uri(DATABASE_URL)
    # Initialize the LLM

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
    # Prepare the prompt with merchant filter inserted

    prompt = BASE_PROMPT.partial(merchant_id=request.merchant_id)

    db_chain = SQLDatabaseChain.from_llm(llm, db, prompt, verbose=True)

    intent = classify_intent(llm, request.query)

    if intent == "sales_query":
        # Run the chain to answer the query
        result = db_chain.invoke(request.query)
        answer = result
    else:
        answer = "I’m here to help with sales data questions. Please ask something related to sales."

    return {"answer": answer}