import streamlit as st
from rag import llm_init, get_answer

# Title for the portfolio chatbot
st.title("Portfolio Chatbot")

# Input area for the user's question
user_question = st.text_input("Ask me anything about my experiences:")

llm, vs = llm_init()

# Button to submit the question
if st.button("Submit") and user_question:
    # Retrieve the answer using your RAG logic
    answer = get_answer(llm, user_question, vs)  # function that wraps RAG code logic
    st.write("Answer:", answer)