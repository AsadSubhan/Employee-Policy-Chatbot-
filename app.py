import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pymysql
from langchain.prompts import PromptTemplate
from system_prompt import get_system_prompt


# Load environment variables
load_dotenv('.env')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


@st.cache_resource
def process_pdf():
    """Read and process the PDF once, cache the vector store."""
    pdfreader = PdfReader('employee_manual.pdf')

    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(chunks, embeddings)

    return document_search

SYSTEM_PROMPT = get_system_prompt()

@st.cache_resource
def load_chain():  
    """Cache the QA chain with a system prompt."""

    prompt = PromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    )

    return load_qa_chain(ChatOpenAI(model="gpt-4o", temperature=0.7), 
                         chain_type="stuff", 
                         prompt=prompt)

@st.cache_resource
def init_connection_pool():
    """Cache connection parameters that will be reused."""
    return {
        "host": os.getenv('DB_HOST'),
        "user": os.getenv('DB_USER'),
        "password": os.getenv('DB_PASS'),
        "port": 3306,
        "database": os.getenv('DB_NAME'),
        "autocommit": True
    }

def get_db_connection():
    """Get a fresh connection using the cached parameters."""
    params = init_connection_pool()
    return pymysql.connect(**params)


# Function to execute a query safely
def execute_query(sql, values=None, fetch=False):
    """Execute a query safely with proper connection handling."""
    db = get_db_connection()
    cursor = db.cursor()
    try:
        db.ping(reconnect=True)  # Ensure connection is active
        if values:
            cursor.execute(sql, values)
        else:
            cursor.execute(sql)
        if fetch:
            return cursor.fetchall()
        db.commit()
    except pymysql.MySQLError as e:
        st.error(f"Database Error: {e}")
    finally:
        cursor.close()
        db.close()  # Ensure connection is closed after query execution


# Cache query results 
@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_query_history():
    """Cache the query results"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql_read = "SELECT `QUERY` FROM DETAILS ORDER BY ID DESC LIMIT 5"
        cursor.execute(sql_read)
        return cursor.fetchall()
    except pymysql.MySQLError as e:
        st.error(f"History fetch error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()  #  close connection to prevent leaks


# Get cached resources
document_search = process_pdf()
chain = load_chain()


def chat_stream():
    st.title("Hello Arpatechians!")
    st.header("Chatbot for Employee Policies")

    # Form to allow submission via Enter
    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input("Please ask a question regarding company policy", key="query_input")
        submitted = st.form_submit_button("Submit")
        st.markdown("""
            <style>
                div.stButton > button:first-child {
                    background-color: red;
                    color: white;
                    border-radius: 5px;
                    border: 1px solid darkred;
                    font-weight: bold;
                }
                div.stButton > button:first-child:hover {
                    background-color: darkred;
                }
            </style>
        """, unsafe_allow_html=True)

    if submitted and query:
        # Search for relevant documents
        docs = document_search.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)

        st.write(response)

        # Store query and response in the database
        sql = "INSERT INTO DETAILS (QUERY, RESPONSE, RATING) VALUES (%s, %s, %s)"
        val = (query, response, "")
        execute_query(sql, val)

        # Clear the history cache to refresh with new query
        get_query_history.clear()

    with st.sidebar:
        st.title("History")
        result = get_query_history( )
        for row in result:
            st.write(":point_right:", row[0])

chat_stream()
