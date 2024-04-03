import streamlit as st
#import pickle
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import dill
import os

#conn = create_engine('mssql+pyodbc://DESKTOP-1B3PJIL/QARetrieval?trusted_connection=yes&driver=ODBC Driver 17 for SQL Server')

# Database connection configuration
host = 'dpg-co5u6lsf7o1s73a7ql50-a'
database = 'qa_y9pi'
user = 'postgre'
password = 'w75OQFSRMPDv2SMrjEsWR8XoLc5d62rL'

# Construct the connection string
conn_str = f"host={host} dbname={database} user={user} password={password}"

#os.environ["GOOGLE_API_KEY"] = "AIzaSyBhJTFGzAMfZo7NZFfDk5J7SHjfdmgHTp4"
api_key = "AIzaSyBhJTFGzAMfZo7NZFfDk5J7SHjfdmgHTp4"

# Path to the combined FAISS index file
combined_faiss_index_path = 'combined_faiss_index.pickle'
# Get the current script directory
#script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the combined FAISS index file in the same folder as the script
#combined_faiss_index_path = os.path.join(script_dir, 'combined_faiss_index.pickle')


# Load the combined FAISS index and texts
with open(combined_faiss_index_path, 'rb') as f:
    combined_faiss_data = dill.load(f)
    combined_faiss_index = combined_faiss_data['combined_index']
    all_texts = combined_faiss_data['all_texts']

# Load the question-answering chain
chain = load_qa_chain(GooglePalm(google_api_key=api_key), chain_type="stuff")

# Streamlit app
st.set_page_config("Question Answering App")

st.title("Question Answering App")

# User input for the question
question = st.text_input("Ask a question:")

# Button to trigger question answering
if st.button("Get Answer"):
    if question:
        # Run the question-answering chain on the combined documents and question
        docs = combined_faiss_index.similarity_search(question)
        answer = chain.run(input_documents=docs, question=question)
        #answer_df = pd.DataFrame({'question': [question],'answer':[answer]})
        #answer_df.to_sql(name="QARetrieval",con = conn,if_exists='append', index = False)
        with psycopg2.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO \"QA retrieval\" (question, answer) VALUES (%s, %s)", (question, answer))
            conn.commit()
        st.success(answer)
    else:
        st.warning("Please enter a question.")

# Additional components or visualizations can be added as needed.
