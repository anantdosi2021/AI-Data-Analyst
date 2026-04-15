import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq  # New Import
from dotenv import load_dotenv
import os

# 1. SETUP: Load Groq API Key
load_dotenv()
# This will look for GROQ_API_KEY in your Streamlit Secrets
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI Business Analyst", layout="wide")
st.title("⚡ AI Business Analyst (Powered by Groq)")

# 2. DATA ANALYTICS: Upload and Process (Same as before)
uploaded_file = st.file_uploader("Upload your Sales CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Data Preview")
        st.dataframe(df.head())
    
    with col2:
        st.write("### Quick Metrics")
        total_sales = df['Sales'].sum()
        avg_sales = df['Sales'].mean()
        st.metric("Total Revenue", f"${total_sales:,.2f}")
        st.metric("Average Sale Value", f"${avg_sales:,.2f}")

    # 3. AGENTIC AI: Context-Aware Chat
    st.divider()
    user_query = st.text_input("Ask the AI Agent about your business trends:")

    if user_query:
        if not api_key:
            st.error("Please add your GROQ_API_KEY to Streamlit Secrets!")
        else:
            try:
                # Initialize Groq with a powerful Llama model
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile", 
                    groq_api_key=api_key,
                    temperature=0.5
                )
                
                data_summary = f"Total sales: {total_sales}, Average: {avg_sales}."
                prompt = f"System: You are a business analyst. Data: {data_summary}. Question: {user_query}"
                
                with st.spinner("Groq is analyzing at lightning speed..."):
                    response = llm.invoke(prompt)
                    st.write("### AI Insight:")
                    st.info(response.content)
            except Exception as e:
                st.error(f"Error connecting to Groq: {str(e)}")
