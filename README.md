# Company Policy Chatbot

This repository contains the code for a chatbot designed to assist employees in querying various policies of the company. The chatbot aims to provide quick responses to employee queries without the need to contact HR or refer to lengthy manuals. It utilizes Python, OpenAI API, Langchain, Streamlit, and MySQL database to facilitate conversation and store interaction history.

## Technologies Used:

- **Python:** The primary programming language used for development.
- **OpenAI API:** Leveraged for natural language processing and generating responses.
- **Langchain:** Utilized for text processing and embedding.
- **Streamlit:** Used to create a user-friendly interface for interacting with the chatbot.
- **MySQL Database:** Employed to store conversation history and user interactions.

## Setup:

1. **Environment Setup:** Ensure you have Python installed on your system.
2. **Dependencies Installation:** Install the required dependencies using `pip install -r requirements.txt`.
3. **Environment Variables:** Set up a `.env` file with the necessary environment variables (e.g., DB_HOST, DB_USER, DB_PASSWORD, OPENAI_API_KEY).
4. **Database Setup:** Create a MySQL database named "mydb" and execute the SQL script provided in the code to create the necessary table.
5. **PDF File:** Ensure you have an employee manual PDF named "employee_manual.pdf" in the root directory.

## Usage:

1. Run the `chat_stream()` function to launch the Streamlit application.
2. Input your query regarding company policy in the text field.
3. Click the "Submit" button to receive a response from the chatbot.
4. Optionally, you can rate the response for future reference.

## Functionality:

- **Query Processing:** The chatbot processes user queries using Langchain for text splitting and embedding.
- **Document Search:** It searches the employee manual PDF for relevant information using FAISS for similarity search.
- **Response Generation:** The chatbot generates responses using the OpenAI API with a temperature setting of 0.7.
- **Conversation History:** User interactions and responses are stored in the MySQL database for future reference.
- **User Interface:** Streamlit provides a user-friendly interface for interacting with the chatbot and viewing conversation history.

## Contributors:

- Asad Subhan


