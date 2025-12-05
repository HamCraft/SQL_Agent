#!/usr/bin/env python3
"""
merchant_cli.py
- Uses your DATABASE_URL env var
- Pulls daily order_total sums from orders.created_at
- Forecasts with Prophet (default 30 days)
- Summarizes forecast using your LLM
"""

from dataclasses import dataclass
import os
import sys
import re
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

# Forecasting imports
from prophet import Prophet
import pandas as pd
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Context (for agent)
@dataclass
class Context:
    db: SQLDatabase

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("Please set DATABASE_URL in your .env file")

# Initialize LLM model
try:
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
except Exception as e:
    raise Exception(f"Failed to initialize model: {str(e)}")

# Initialize SQLDatabase and toolkit (keeps agent available)
db = SQLDatabase.from_uri(DATABASE_URL)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

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

# Create agent (still available for general Q->SQL reasoning)
agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
    context_schema=Context,
)

# Forbidden keywords for user questions (safety)
FORBIDDEN = ["schema", "table", "tables", "columns", "database structure", "id", "tab", "leak", "key"]

# --- Database helper: direct safe read (aggregated time series) ---
# NOTE: This uses direct DB access via SQLAlchemy + pandas to build a
# time series for Prophet. We aggregate by date to produce stable daily series.
def get_daily_time_series(date_col="created_at", value_col="order_total", merchant_id=None):
    """
    Returns DataFrame with columns ['ds', 'y'] where ds is date (datetime.date)
    and y is numeric aggregated value (sum) per day.
    merchant_id: int or None -> filters by merchant if provided
    """
    engine = create_engine(DATABASE_URL)
    # Build a parameterized query (safe)
    if merchant_id is not None:
        q = text(
            f"""
            SELECT DATE({date_col}) AS ds,
                   SUM({value_col}) AS y
            FROM orders
            WHERE merchant_id = :merchant_id
            GROUP BY DATE({date_col})
            ORDER BY DATE({date_col}) ASC
            """
        )
        df = pd.read_sql_query(q, engine, params={"merchant_id": merchant_id})
    else:
        q = text(
            f"""
            SELECT DATE({date_col}) AS ds,
                   SUM({value_col}) AS y
            FROM orders
            GROUP BY DATE({date_col})
            ORDER BY DATE({date_col}) ASC
            """
        )
        df = pd.read_sql_query(q, engine)

    # Ensure proper types
    if df.empty:
        return df
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    return df


# --- Prophet forecasting ---
def prophet_forecast(df, periods=30, freq="D"):
    """
    df: DataFrame with columns ['ds','y']
    periods: number of future periods (days)
    returns: forecast DataFrame with ds, yhat, yhat_lower, yhat_upper
    """
    if df is None or df.empty:
        return {"error": "No historical data available for forecasting."}

    # Ensure valid order and no duplicates
    df = df.sort_values("ds").drop_duplicates(subset=["ds"])
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
    # Convert ds to isoformat strings for serialization
    out["ds"] = out["ds"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


# --- Utility: extract merchant id from question if present ---
def extract_merchant_id(question: str):
    """
    Tries to extract patterns like 'merchant 123' or 'merchant_id 123'
    Returns int or None
    """
    # common patterns
    m = re.search(r"merchant[_\s]?id\s*[:=]?\s*(\d+)", question, re.I)
    if not m:
        m = re.search(r"merchant\s+(\d+)", question, re.I)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


# --- Main query runner (integrates agent and forecasting) ---
def run_query(question: str):
    q = question.strip()

    # Safety: forbid schema-level requests
    if any(word in q.lower() for word in FORBIDDEN):
        return "Apologies, I can't provide details on that. Anything else you would like to ask?"

    # If this is a forecast request, handle with Prophet pipeline
    if any(k in q.lower() for k in ["forecast", "predict", "projection", "projection", "future", "predicting"]):
        merchant_id = extract_merchant_id(q)
        df = get_daily_time_series(date_col="created_at", value_col="order_total", merchant_id=merchant_id)
        if df.empty:
            if merchant_id:
                return f"No historical data found for merchant {merchant_id} to produce a forecast."
            else:
                return "No historical data found to produce a forecast."

        # Decide periods: try to extract number of days from question
        periods = 30  # default
        m_days = re.search(r"next\s+(\d+)\s+days", q, re.I)
        if not m_days:
            m_days = re.search(r"(\d+)\s+day(s)?\s+forecast", q, re.I)
        if m_days:
            try:
                periods = int(m_days.group(1))
            except Exception:
                periods = 30

        forecast_records = prophet_forecast(df, periods=periods, freq="D")
        if isinstance(forecast_records, dict) and forecast_records.get("error"):
            return forecast_records["error"]

        # short numeric summary (Totals / trend)
        # Build quick summary for the LLM to explain
        sample_summary_text = (
            f"Here are {periods} daily forecast rows (ds, yhat, yhat_lower, yhat_upper):\n"
            + "\n".join([f"{r['ds']}: {r['yhat']:.2f} (±{(r['yhat']-r['yhat_lower']):.2f})" for r in forecast_records[:5]])
            + "\n\nPlease provide a short plain-language summary of the expected trend and any notable observations."
        )

        # Ask LLM to summarize (we use a short prompt)
        try:
            summary_resp = model.invoke(sample_summary_text)
            summary_text = summary_resp if isinstance(summary_resp, str) else str(summary_resp)
        except Exception:
            summary_text = "Forecast generated. (LLM summary unavailable.)"

        # Return LLM summary + tabular forecast (first & last few rows)
        # Provide a concise text payload to CLI. The user can ask for full data if needed.
        first_rows = forecast_records[:5]
        last_rows = forecast_records[-5:]
        result_text = f"{summary_text}\n\nFirst {len(first_rows)} forecast days:\n"
        for r in first_rows:
            result_text += f"{r['ds']}: yhat={r['yhat']:.2f}, lower={r['yhat_lower']:.2f}, upper={r['yhat_upper']:.2f}\n"
        result_text += "\nLast 5 forecast days:\n"
        for r in last_rows:
            result_text += f"{r['ds']}: yhat={r['yhat']:.2f}, lower={r['yhat_lower']:.2f}, upper={r['yhat_upper']:.2f}\n"

        # Also include an optional instruction to retrieve the full forecast as JSON if the user wants it
        result_text += "\n(To get the full forecast output as JSON, ask: 'show full forecast json')"

        # Store last forecast in memory so "show full forecast json" can return it
        run_query._last_full_forecast = forecast_records  # attach to function

        return result_text

    # If user asks to show full forecast previously generated
    if re.search(r"show\s+full\s+forecast", q, re.I) and hasattr(run_query, "_last_full_forecast"):
        import json
        return json.dumps(run_query._last_full_forecast, indent=2)

    # Otherwise, fall back to the agent to answer non-forecast questions
    final_answer = ""
    for step in agent.stream(
        {"messages": [{"role": "user", "content": q}]},
        {"configurable": {"thread_id": "0103"}},
        context=Context(db=db),
        stream_mode="values",
    ):
        final_answer = step["messages"][-1].text

    return final_answer


# --- CLI interactive mode ---
def interactive_mode():
    print("Merchant Intelligence Agent CLI")
    print("Type 'exit' to quit.\n")
    print("Examples:")
    print("  forecast next 30 days for merchant 2001")
    print("  forecast 14 day sales")
    print("  show full forecast json\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
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
        interactive_mode()
