import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from fastapi_cache import caches

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# Load environment variables
load_dotenv()

# Read full DATABASE_URL from .env
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("Please set DATABASE_URL in your .env file")

# Initialize the FastAPI app
app = FastAPI(title="Postgres SQL Agent API")

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis cache setup (optional, for caching common queries)
cache = caches.get("default")


# DatabaseAgent class to handle the model, toolkit, and agent setup
class DatabaseAgent:
    def __init__(self, database_url: str, model_name: str = "gemini-2.5-flash"):
        try:
            self.db = SQLDatabase.from_uri(database_url)
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise Exception(f"Failed to connect to database: {e}")

        try:
            self.model = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise Exception(f"Failed to initialize model: {e}")

        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.model)
        self.tools = self.toolkit.get_tools()

        # System prompt to guide the agent
        self.system_prompt = """
        Answer questions about the data only.
        Do NOT reveal table names, column names, or schema.
        Limit query results to 5 rows max.
        Do NOT run INSERT, UPDATE, DELETE, DROP, or other DML.
        If asked about structure, politely refuse.
        Answer concisely.
        DO Not talk about irrelevant stuff.
        """

        # Create the agent with the model and tools
        self.agent = create_agent(self.model, self.tools, system_prompt=self.system_prompt)

    def ask(self, question: str) -> str:
        """
        Method to process the user's question through the AI agent
        """
        logger.info(f"Processing question: {question}")
        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            return step["messages"][-1].text


# Initialize the DatabaseAgent
agent = DatabaseAgent(DATABASE_URL)


# Request model for the API
class QueryRequest(BaseModel):
    question: str


# API endpoint to handle user questions
@app.post("/ask", summary="Ask a Question to the Database", description="Send a question to the AI agent about the database.")
async def ask_question(request: QueryRequest, api_key: str = Header(...)):
    """
    Accepts a question about the database and returns the AI model's answer. 
    The model will not reveal database structure or sensitive details.
    """
    # Check for valid API key (simple authentication)
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    question = request.question.strip()

    # Simple sanitization: block questions explicitly asking for schema or table names
    forbidden_keywords = ["schema", "table", "tables", "columns", "database structure", "base", "id", "--", ";", "'"]
    if any(word in question.lower() for word in forbidden_keywords):
        return {"answer": "Apologies, I can't provide details on that. Is there anything else I can assist you with?"}

    # Check if the question is cached
    cached_answer = await cache.get(question)
    if cached_answer:
        logger.info(f"Returning cached answer for: {question}")
        return {"answer": cached_answer}

    try:
        # Get the final answer from the agent
        final_answer = agent.ask(question)

        # Cache the answer (optional, for repeated queries)
        await cache.set(question, final_answer, ttl=3600)

        logger.info(f"Final answer for '{question}': {final_answer}")
        return {"answer": final_answer}
    except Exception as e:
        logger.error(f"Error processing question '{question}': {e}")
        return {"answer": "Sorry, there was an error processing your question."}


# This is important for Vercel (or other environments) to run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

