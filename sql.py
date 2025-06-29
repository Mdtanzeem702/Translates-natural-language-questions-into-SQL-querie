from dotenv import load_dotenv
load_dotenv() 

import streamlit as st
import os
import sqlite3

from langchain_groq import ChatGroq


from langchain_core.prompts import ChatPromptTemplate



os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
api_key= os.environ["GROQ_API_KEY"]


# Define the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6,
    max_retries=2,
    api_key=api_key,
    max_tokens=200,
)
def get_gemini_response(llm, question, prompt):
    # Ensure prompt is a string
    if isinstance(prompt, list):
        prompt = prompt[0]

    # Build prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("user", "{question}")
    ])

    # Format the messages for the LLM
    formatted_messages = prompt_template.format_messages(question=question)

    # Debug: print the messages being sent
    for msg in formatted_messages:
        print(f"{msg.type.upper()}: {msg.content}")

    # Get LLM response
    response = llm.invoke(formatted_messages)

    return response.content


## Function To retrieve query from the database

def read_sql_query(sql,db):
    conn=sqlite3.connect(db)
    cur=conn.cursor()
    cur.execute(sql)
    rows=cur.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows

## Define Your Prompt
prompt=[
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name STUDENT and has the following columns - NAME, CLASS, 
    SECTION,MARKS \n\nFor example,\nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM STUDENT ;
    \nExample 2 - Tell me all the students studying in Data Science class?, 
    the SQL command will be something like this SELECT * FROM STUDENT 
    where CLASS="Data Science"; 
    also the sql code should not have ``` in beginning or end and sql word in output

    """
]

## Streamlit App

st.set_page_config(page_title="I can Retrieve Any SQL query")
st.header("Gemini App To Retrieve SQL Data")

question=st.text_input("Input: ",key="input")

submit=st.button("Ask the question")

# if submit is clicked
if submit:
    response=get_gemini_response(llm,question,prompt)
    st.header(f"The Generated SQL Query is: ")
    st.subheader(response)
    response=read_sql_query(response,"student.db")
    st.subheader("The Response is")
    for row in response:
        st.header(row)








