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
        self.db = SQLDatabase.from_uri(r"sqlite:///C:\Users\MDTAN\OneDrive\Desktop\Msc Project UOR\Text_to_SQL_Project\NeuroQuery_AI_Driven_Database_Assistant\Project_Code\my_database.db")
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()

        print(self.db.dialect)
        print(self.db.get_usable_table_names())

        # embeddings = GoogleGenerativeAIEmbeddings(
        #     model="models/embedding-001",
        #     task_type="retrieval_document",  # Add this parameter
        #     client_options={"api_key": "your-api-key"} )
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
        """
        response = None
        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
            
        response = step

        print(f"Received question: {question}")
        
        if response:
            print(response["messages"][-1].content)
            return {
                "query": response["messages"][-1].content,
                "answer": response["messages"][-1].content
            }
        return {
            "query": "No query generated",
            "answer": "No answer generated"
        }
    

# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import sqlite3

# load_dotenv()
# llm=ChatGroq(model='llama-3.1-8b-instant',temperature=0,max_tokens=200)


# def get_database_info():
    
#     try:
#         conn=sqlite3.Connection(r"D:\text_to_sql_project\Tanzeem_Project\my_database.db")
#         cursor = conn.cursor()

#         # Get all user-defined table names
#         cursor.execute("""
#             SELECT name FROM sqlite_master
#             WHERE type='table' AND name NOT LIKE 'sqlite_%';
#         """)
#         tables = [row[0] for row in cursor.fetchall()]

#         result = {}
#         for table in tables:
#             cursor.execute(f"PRAGMA table_info({table})")
#             columns = [col[1] for col in cursor.fetchall()]  # col[1] is column name
#             result[table] = columns

#         conn.close()
#         return result
#     except Exception as e:
#         print('An Execption occured!!',e)
#         return "None"

# def generate_schema_description(schema_dict):
#     schema_lines = []
#     for table, columns in schema_dict.items():
#         column_list = ', '.join(columns)
#         schema_lines.append(f"- {table.upper()} table with columns: {column_list}")
#     return "\n".join(schema_lines)


# def get_template_prompt(question):


#     database_info=get_database_info()

#     if database_info != "None":
#         schema_description = generate_schema_description(database_info)
#         print(schema_description)
#         messages=[
#             ("system",    f"""
#                             You are an expert in converting English questions to SQL queries!
                            
#                             The database has the following tables and columns:
#                             {schema_description}

#                             Always generate correct SQL queries for the SQLite database above.

#                             Example 1 – How many entries are there in STUDENT?
#                             → SELECT COUNT(*) FROM STUDENT;

#                             Example 2 – Show all teachers who teach Physics.
#                             → SELECT * FROM TEACHER WHERE subject = "Physics";

#                             Rules:
#                             - Don't include the word "SQL" or use triple backticks in your output.
#                             - Only return the raw SQL query.
#                             """),
#             ("human","Answer the user's query: {question}")
#         ]
#         # print("messages: ",messages)
#         prompt=ChatPromptTemplate.from_messages(messages)
#         final_prompt=prompt.invoke({"question": {question}})
#         # print("final_prompt: ",final_prompt)

#         return final_prompt
#     else:
#         messages=[
#             ("system","you are a helpful assistant, here you'll just notify the user about that an error occured!"),
#             ("human","Notify the user: {question}")
#         ]
#         # print("messages: ",messages)
#         prompt=ChatPromptTemplate.from_messages(messages)
#         error_final_prompt=prompt.invoke({"question": {question}})
#         # print("final_prompt: ",final_prompt)

#         return error_final_prompt


# def handle_question(question,llm=llm):
#     """
#     Replace this logic with actual LLM or agent processing.
#     """
#     print(f"Received question: {question}")

#     formated_prompt=get_template_prompt(question=question)
    
#     response=llm.invoke(formated_prompt)
#     # Fake SQL and answer (replace this with real logic)
#     print("\n\n\n respone: ",response.content)
#     fake_sql = response.content
#     fake_answer = response.content

#     return {
#         "query": fake_sql,
#         "answer": fake_answer
#     }
