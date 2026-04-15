import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv() # Loads your API Key from a .env file

st.title("🤖 AI Business Analyst")

# 1. DATA ANALYTICS: Upload and Process
uploaded_file = st.file_uploader("Upload your Sales CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    
    # Simple calculation
    total_sales = df['Sales'].sum()
    st.metric("Total Sales", f"${total_sales:,.2f}")

    # 2. AGENTIC AI: Ask questions about the data
    user_query = st.text_input("Ask me anything about your data:")
    if user_query:
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        # In a real RAG, you'd feed the 'df' summary here
        response = llm.invoke(f"Based on these sales of {total_sales}, answer: {user_query}")
        st.write("### AI Response:", response.content)