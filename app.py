import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# 1. SETUP: Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="AI Business Analyst", layout="wide")
st.title("🤖 AI Business Analyst (Powered by Gemini)")

# 2. DATA ANALYTICS: Upload and Process
uploaded_file = st.file_uploader("Upload your Sales CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Create two columns for the UI
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
            st.error("Please add your GOOGLE_API_KEY to the settings!")
        else:
            # We initialize Gemini here
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)
            
            # We give the AI a summary of the data so it can answer accurately (RAG logic)
            data_summary = f"The total sales are {total_sales} and average sale is {avg_sales}."
            prompt = f"System: You are a business expert. Data: {data_summary}. Question: {user_query}"
            
            with st.spinner("Analyzing..."):
                response = llm.invoke(prompt)
                st.write("### AI Insight:")
                st.info(response.content)
