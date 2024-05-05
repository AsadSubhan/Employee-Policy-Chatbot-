import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st
import pymysql

load_dotenv('.env')

db = pymysql.connect(host=os.getenv('DB_HOST'), 
                     user=os.getenv('DB_USER'),
                     password=os.getenv('DB_PASSWORD'),
                     port= 3306,
                     database= "mydb")

cursor = db.cursor()

# cursor.execute(""" 
#    CREATE TABLE DETAILS(
#    ID INT NOT NULL AUTO_INCREMENT,
#    QUERY VARCHAR(1000) NOT NULL, 
#    RESPONSE TEXT NOT NULL, 
#    RATING VARCHAR(25), 
#    PRIMARY KEY(ID)
# );
# """)
# print("connected")



os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

pdfreader = PdfReader('arpatech_employee_manual.pdf')

raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200
)

chunks = splitter.split_text(raw_text)
print(len(chunks))

embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(chunks, embeddings)

chain = load_qa_chain(OpenAI(temperature = 0.7), chain_type ="stuff")


def chat_stream():
    st.title("Hello Arpatechians!")
    st.header("Chatbot for Employee Policies")

    query = st.text_input("Please ask a question regarding company policy")
    docs = document_search.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)

    press = st.button("Submit", type="primary")
    rating = ""

    if press == True:
        st.write(response)
    
        sql = "INSERT INTO DETAILS (QUERY, RESPONSE, RATING) VALUES (%s, %s, %s)"
        val = (query, response, rating)
        cursor.execute(sql, val)
        db.commit()
    
    sql_read = "SELECT `QUERY` FROM DETAILS ORDER BY ID DESC LIMIT 5"
    cursor.execute(sql_read)

    result = cursor.fetchall()

    with st.sidebar:
        st.title("History")
        for row in result:
            st.write(":point_right:",row[0])

chat_stream()