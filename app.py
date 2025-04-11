import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
import streamlit as st
import pymysql
from langchain.prompts import PromptTemplate
from system_prompt import get_system_prompt
from langdetect import detect, LangDetectException
import re


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

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    document_search = FAISS.from_texts(chunks, embeddings)

    return document_search

@st.cache_resource
def process_roman_words():
    with open("roman_urdu_cleaned_list.txt", "r", encoding="utf-8") as file_words:
        content_roman = file_words.read()
        roman_words_list = [word.strip() for word in content_roman.split(',') if word.strip()]

    return roman_words_list


def detect_language(text):
    # Special handling for very short text
    if len(text.split()) <= 3:  # For texts with 3 or fewer words
        # Check if text contains only English alphabet characters and common punctuation
        if re.match(r'^[a-zA-Z\s.,!?-]+$', text):
            return "Query is in English, so you should respond in English as well."
    
    # For longer text, try language detection
    try:
        # Use langdetect with error handling
        detected_lang = detect(text)
        
        # Higher confidence in English detection
        if detected_lang == "en":
            return "Query is in English, so you should respond in English as well."
        
        # Check for Roman Urdu words (using whole word matching)
        text_lower = text.lower()
        words_in_text = set(text_lower.split())
        roman_urdu_words_in_text = words_in_text.intersection(set(process_roman_words()))
        
        if roman_urdu_words_in_text:
            matched_words = list(roman_urdu_words_in_text)
            # Optionally log which words matched
            # print(f"Matched Roman Urdu words: {matched_words}")
            return "Query is in Roman Urdu, so you should respond in Roman Urdu as well."
        
        # Double-check English: Common words that strongly indicate English
        english_indicators = {"the", "is", "are", "and", "or", "but", "a", "an", "in", "on", "at", 
                             "of", "for", "to", "with", "by", "from", "as", "that", "this", "these", 
                             "those", "my", "your", "our", "their", "his", "her", "its"}
        
        if words_in_text.intersection(english_indicators):
            return "Query is in English, so you should respond in English as well."
        
        if detected_lang == "ur":
            return "Query is in Urdu, so you should respond in Urdu as well."
        
        # Fallback to detected language if we're still unsure
        return f"Query is in {detected_lang}, so you should respond in {detected_lang} as well."
        
    except LangDetectException:
        # Fallback to English if language detection fails
        return "Unable to detect language confidently. Defaulting to English response."


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
        db.close()  # Ensure connection is closed after execution


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
        # Detect the language of the query
        language_info = detect_language(query)

        # Dynamically generate the system prompt based on detected language
        SYSTEM_PROMPT = get_system_prompt(language_info)

        # Build the prompt template with the updated SYSTEM_PROMPT
        prompt = PromptTemplate.from_template(
            SYSTEM_PROMPT + "\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
        )

        # Initialize the QA chain with the new prompt
        chain = load_qa_chain(ChatOpenAI(model="gpt-4o", temperature=0.7), 
                            chain_type="stuff", 
                            prompt=prompt)

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
