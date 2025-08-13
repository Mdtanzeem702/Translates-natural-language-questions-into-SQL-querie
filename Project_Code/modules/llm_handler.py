from urllib import response
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from typing_extensions import TypedDict

import ast
import re

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


class my_llm_response:
    def __init__(self):
        load_dotenv()
        
        self.system_message = """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        Then you should query the schema of the most relevant tables.
        """.format(
            dialect="SQLite",
            top_k=5,
        )

        self.description = (
            "Use to look up values to filter on. Input is an approximate spelling "
            "of the proper noun, output is valid proper nouns. Use the noun most "
            "similar to the search."
        )

        self.suffix = (
            "If you need to filter on a proper noun like a Name, you must ALWAYS first look up "
            "the filter value using the 'search_proper_nouns' tool! Do not try to "
            "guess at the proper name - use this function to find similar ones."
        )

        self.llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        self.db = SQLDatabase.from_uri(r"sqlite:///C:/Users/MDTAN/OneDrive/Desktop/Msc Project UOR/Text_to_SQL_Project/NeuroQuery_AI_Driven_Database_Assistant/Project_Code/Student_Project_Management_DB.db")
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()

        print(self.db.dialect)
        print(self.db.get_usable_table_names())

        vector_store = InMemoryVectorStore(embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        retriever_tool = create_retriever_tool(
            retriever,
            name="search_proper_nouns",
            description=self.description,
        )

        system = f"{self.system_message}\n\n{self.suffix}"
        self.tools.append(retriever_tool)
        self.agent = create_react_agent(self.llm, self.tools, prompt=system)
    
    def query_as_list(self, db, query):
        res = db.run(query)
        res = [el for sub in ast.literal_eval(res) for el in sub if el]
        res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        return list(set(res))

    def handle_question(self, question):
        """
        Process the question using the LLM agent and return the response.
        Returns a dict with 'query' (SQL if found) and 'answer' (final response).
        """
        response = None
        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
            response = step  # Keep updating response until last step

        if not response:
            return {"query": None, "answer": "No response from agent"}

        # Extract all SQL queries from the conversation
        sql_queries = self.extract_all_sql_queries(response)
        
        # Get the most recent valid SQL query (if any)
        final_sql_query = sql_queries[-1] if sql_queries else None
        
        return {
            "query": final_sql_query,
            "answer": response["messages"][-1].content
        }

    def extract_all_sql_queries(self, step):
        """
        Extracts all SQL queries from the agent's conversation steps.
        Returns a list of queries in chronological order.
        Handles cases where:
        - Query appears multiple times (returns all)
        - Query appears in different tools (query_checker, sql_db_query)
        - No query is found (returns empty list)
        """
        queries = []
        
        for message in step.get("messages", []):
            # Check for tool calls in the message
            tool_calls = getattr(message, 'tool_calls', []) or message.additional_kwargs.get('tool_calls', [])
            
            for call in tool_calls:
                # Handle both direct SQL queries and query checkers
                if call.get("name") in ["sql_db_query", "sql_db_query_checker"]:
                    query = call.get("args", {}).get("query")
                    if query:
                        queries.append(query)
        
        return queries
