from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_classic import hub
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(override=True)

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("Please set DATABASE_URL in your .env file")
db = SQLDatabase.from_uri(DATABASE_URL)

# Pull the SQL query prompt from LangChain Hub
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")


def write_query(question: str):
    """Generate SQL query from the user's question."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": question,
        }
    )
    response = llm.invoke(prompt.to_string())

    extraction_prompt = """
    Please extract the SQL query from the following text and return only the SQL query without any additional characters or formatting:

    {response}

    SQL Query:
    """
    # Format the prompt with the actual response
    prompt = extraction_prompt.format(response=response)
    # Invoke the language model with the prompt
    parsed_query = llm.invoke(prompt)
    return parsed_query.content

def execute_query(query: str):
    """Execute the SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return execute_query_tool.invoke(query)

def generate_answer(question: str, query: str, result: str):
    """Generate an answer using the query results."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {question}\n'
        f'SQL Query: {query}\n'
        f'SQL Result: {result}'
    )
    response = llm.invoke(prompt)
    return response.content

question = "What is the staus for this order"
query = write_query(question)
result = execute_query(query)
answer = generate_answer(question,query, result )
print(answer)