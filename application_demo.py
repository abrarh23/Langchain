import os
from langchain_community.utilities import SQLDatabase

from flask import Flask, request, jsonify
from dotenv import load_dotenv
load_dotenv()


#===================================================
#=============== Load the Database =================
#===================================================


DATABASE_USER = os.environ.get('DATABASE_USER')
DATABASE_PASSWORD = os.environ.get('DATABASE_PASSWORD')
DATABASE_HOST = os.environ.get('DATABASE_HOST')
DATABASE_NAME = os.environ.get('DATABASE_NAME')
BASE_URL_OF_SQL_AGENT_MODEL = os.environ.get('BASE_URL_OF_SQL_AGENT_MODEL')


#===================================================
#=============== Load the Database =================
#===================================================


uri = f'mysql+pymysql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_NAME}'
db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=10)

# print("abcd->>>>>.",db.get_table_info())

#===================================================
#=============== Create a SQL Chain =================
#===================================================

from langchain_core.prompts import ChatPromptTemplate

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""

prompt = ChatPromptTemplate.from_template(template)

def get_schema(_):
    schema = db.get_table_info()
    return schema


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="hf.co/defog/sqlcoder-7b-2:latest",
    base_url=BASE_URL_OF_SQL_AGENT_MODEL,
    temperature=0
)

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


#===================================================
#============== Create the full Chain ==============
#===================================================


template_2 = """Use the given table schema, user question, and SQL query as context, but generate a response **only based on the SQL query output**. Do not include explanations about the schema or the query itself. Answer in a natural language format.

Table Schema (for context):  
{schema}

User Question:  
{question}

SQL Query (for reference):  
{query}

SQL Query Output (use only this to generate the response):  
{response}

Generate a concise and clear answer based on the SQL query output."""


prompt_response = ChatPromptTemplate.from_template(template_2)


def run_query(query):
    return db.run(query)


llm_2 = ChatOllama(
    model="llama3.1",
    base_url=BASE_URL_OF_SQL_AGENT_MODEL,
    temperature=0
)


full_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt_response
    | llm_2
)


application = Flask(__name__)


@application.route('/test-url', methods=['GET'])
def test_url():
    return jsonify({"message": "URL accessed successfully!"})


@application.route('/query', methods=['POST'])
def query_db():
    try:
        user_prompt = request.json["prompt"]
    
        try:


            sql_query = None

            model_response = sql_chain.invoke({"question": user_prompt})
            # response = {"input": user_prompt, "output": "Nothing found from Database"}
            print("\n", "model_response-->>>>>", model_response)

            if "sql" in model_response:
                sql_query = model_response.split("```")[1].split("sql")[1].strip()
                print("\n", "sql_query-->>>>>", sql_query)
            else:
                sql_query = model_response.strip()
            

            db_response = run_query(sql_query)
            print("\n", "db_response-->>>>>", str(db_response))

            final_response  = full_chain.invoke({"question": user_prompt, "query": sql_query, "response": db_response})
            print("\n", "final_response-->>>>>", str(final_response))
            output = final_response.content
            response = {"input": user_prompt, "output": str(output)}
            return jsonify({"response": response}), 200

        except Exception as e:
            response = {"input": user_prompt, "output": "Nothing found from Database"}
            return jsonify({"response": response}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    application.run(debug=True)