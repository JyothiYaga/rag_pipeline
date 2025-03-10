# web_ui/app.py
import streamlit as st
import requests

st.title("RAG PDF Query System")

# Upload PDF
st.subheader("Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/upload_pdf/", files=files)
    st.write(response.json())

# Query
st.subheader("Ask a Question")
question = st.text_input("Enter your question")
if st.button("Search"):
    try:
        response = requests.get(f"http://127.0.0.1:8000/query/?question={question}")
        response.raise_for_status()  # Raise exception for non-200 responses
        st.write(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
    except ValueError as e:
        st.error(f"Error parsing response: {str(e)}")
