import os
from dotenv import load_dotenv
import pymysql

load_dotenv('.env')

db = pymysql.connect(host=os.getenv('DB_HOST'), 
                     user=os.getenv('DB_USER'), 
                     password=os.getenv('DB_PASS'), 
                     port=3306, 
                     database='mydb')

cursor = db.cursor()

print("Connected")

cursor.execute("""
    CREATE TABLE DETAILS(
    ID INT NOT NULL AUTO_INCREMENT,
    QUERY VARCHAR(1000) NOT NULL, 
    RESPONSE TEXT NOT NULL,
    RATING VARCHAR(25),
    PRIMARY KEY(ID)
    );
""")

print("Table Created")
