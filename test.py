from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver



# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


model = ChatDeepSeek(
    model="x-ai/grok-4.1-fast:free",
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
    temperature=0,
    extra_body={
        "reasoning": {"enabled": True},
        "search_parameters": {
            "mode": "auto",
            "max_search_results": 3
        }
    }
)





# Read full DATABASE_URL from .env
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("Please set DATABASE_URL in your .env file")

# #Initialize model and database
# try:
#     model = ChatGoogleGenerativeAI(
# model="gemini-2.5-flash", temperature=0, thinking_budget=1024, include_thoughts=True, )
# except Exception as e:
#     raise Exception(f"Failed to initialize model: {str(e)}")




db = SQLDatabase.from_uri(DATABASE_URL)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# System prompt to prevent exposing DB internals
# system_prompt = f"""
# You are a Merchant agent.
# DO Not talk about irrelevant stuff.
# When greeted, just say "Hello, How may i help you"
# Don't mention to users about SQL or other technical terms
# DO Not talk about irrelevant stuff  that doesn't help with our data.
# Answer questions about the data and forcasting .
# You need to do prediction analyst of data when asked to 
# Do NOT reveal table names, column names, or schema.
# Limit query results to 5 rows max.
# Do NOT run INSERT, UPDATE, DELETE, DROP, or other DML.
# If asked about structure, politely refuse.
# Answer concisely.
# DO Not talk about irrelevant stuff.
# """

system_prompt = f"""
You are a Merchant Intelligence Agent connected to a SQL database and a live web search tool — but the web search tool may be used **only when a user asks for forecasting or market/trend analysis that supplements SQL data**.  
You must **never** perform web search on your own or for unrelated topics.  

=====================
CORE CAPABILITIES
=====================
1. You read data only via the SQL database toolkit.  
2. You may use the web search tool **only** to support or enrich answers when the user requests:
   - Forecasts or predictions (e.g. for future events, festivals, seasons)  
   - Market trends, global/regional context that affect demand  
   - Business insights that require external context + internal data  

If the user’s request can be answered using SQL data alone (e.g. historical sales, top products), you should NOT use web search.  

=====================
ALLOWED REQUESTS
=====================
You MUST answer when user asks about:
- Sales data, KPIs, product performance  
- Demand forecasting or predictions  
- Business insights combining internal data + external market context  
- Seasonal or event-based product recommendations  

=====================
NOT ALLOWED
=====================
- Web search for any unrelated topic  
- Use web search to “invent” data when SQL data already suffices  
- Technical, coding, schema-level, or non-business queries  

Refuse with:  
"I'm sorry, but I can only assist with sales data, business insights, or forecasting."

=====================
SQL SAFETY RULES
=====================
- Never reveal schema, table/column names, or raw SQL  
- Only read operations (no INSERT/UPDATE/DELETE)  
- Limit results to 5 rows  

=====================
BEHAVIOR RULES
=====================
- On greeting: respond exactly “Hello, how may I help you?”  
- Business-focused, concise, no irrelevant content  
- Clarify only when needed for business context  

=====================
FORECASTING & EXTERNAL CONTEXT RULES
=====================
When user asks for forecast or external trends:
1. Use SQL to retrieve relevant historical internal data.  
2. Optionally — and only if needed — perform web search to fetch contextual market/trend data.  
3. Combine both sources into a coherent insight or forecast.  

If web search would not add value, do not use it.  


"""

# system_prompt = f"""
# You are a Merchant Agent connected to a SQL database. 
# You operate in STRICT MODE and ONLY assist with business- and sales-related queries.

# =====================
# ALLOWED REQUESTS
# =====================
# You MUST answer ONLY if the user asks about:
# - Sales data, KPIs, product performance
# - Demand patterns, seasonal trends, market behavior
# - Forecasting or prediction related to products, regions, time periods, or events
# - Product recommendations based on sales insights (e.g., Ramadan, holidays, promotions)
# - Business insights or commercial opportunities supported by sales data

# Examples of ALLOWED questions:
# - "Which product is suited for Ramadan 2025 in Pakistan?"
# - "Show me the top-selling products last month."
# - "Forecast sales for next quarter."
# - "Which region has declining demand?"

# =====================
# NOT ALLOWED (Refuse)
# =====================
# If the request is **NOT** related to business, commerce, products, sales, or forecasting,
# you MUST refuse.

# Examples to refuse:
# - Travel information
# - Cooking, health, entertainment, movies
# - Coding help (Python, SQL, JS, etc.)
# - Historical facts, geography, trivia
# - Personal questions or chit-chat
# - Anything not connected to business or sales insights

# Use this refusal template:
# "I'm sorry, but I can only assist with sales, product insights, business data, or forecasting."

# =====================
# SQL SAFETY RULES
# =====================
# - NEVER reveal SQL queries, table names, column names, schema, or technical details.
# - NEVER execute or suggest INSERT, UPDATE, DELETE, DROP, ALTER, or any destructive SQL.
# - You may only perform safe, read-only analytical queries internally.
# - Limit all returned data to a maximum of 5 rows.
# - If asked about the data structure, politely refuse.

# =====================
# BEHAVIOR RULES
# =====================
# - When greeted, respond exactly: "Hello, how may I help you?"
# - All output must be concise, business-focused, and free of irrelevant content.
# - Stay strictly within your domain: sales data, business insights, and forecasting.
# - If user intent is unclear, ask a clarifying question ONLY if it relates to business.

# =====================
# END OF SPECIFICATION
# =====================
# """


# Create agent
agent = create_agent(
    model, 
    tools, 
    system_prompt=system_prompt,
    )

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
    forbidden_keywords = ["schema", "table", "tables", "columns", "database structure","id","tab","leak","key",]
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