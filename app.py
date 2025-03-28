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

# Load environment variables
load_dotenv('.env')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


@st.cache_resource
def process_pdf():
    """Read and process the PDF once, cache the vector store."""
    pdfreader = PdfReader('arpatech_employee_manual.pdf')

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


@st.cache_resource
def load_chain():
    """Cache the QA chain."""
    return load_qa_chain(ChatOpenAI(model="gpt-4-turbo", temperature=0.7), chain_type="stuff")


# Function to get a new database connection
def get_db_connection():
    """Get a new database connection (not cached)."""
    return pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS'),
        port=3306,
        database="mydb",
        autocommit=True  # Ensures each query is committed immediately
    )


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

    # Fetch last 5 queries for history
    sql_read = "SELECT `QUERY` FROM DETAILS ORDER BY ID DESC LIMIT 5"
    result = execute_query(sql_read, fetch=True)

    with st.sidebar:
        st.title("History")
        for row in result:
            st.write(":point_right:", row[0])


chat_stream()
