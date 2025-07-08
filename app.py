# 📂 app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import google.generativeai as genai
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from src.model_classifier import classify_text
import time

load_dotenv()

# --------------------------------------------------------------------------------------------------------------
# Setup Google Gemini API
# --------------------------------------------------------------------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# --------------------------------------------------------------------------------------------------------------
# App Config
# --------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="InsightNation Dashboard", layout="wide")
st.title("InsightNation - Government Data Analytics Platform")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

# --------------------------------------------------------------------------------------------------------------
# File Upload Section
# --------------------------------------------------------------------------------------------------------------
def upload_dataset():
    uploaded_file = st.file_uploader("Upload your cleaned citizen feedback CSV:", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")
        st.dataframe(df.head(20))
        st.session_state["df"] = df



# --------------------------------------------------------------------------------------------------------------
# Citizen Feedback Insights
# ---------------------------------------------------------------------------------------------------------------
def citizen_feedback_insights():
    df = st.session_state.get("df")
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    if st.button("Generate Feedback Insights"):
        st.subheader("Key Feedback Statistics")
        
        # avg_scores = {
        #     "Toilet Cleanliness": df["toilet_cleanliness"].mean(),
        #     "Transport Satisfaction": df["transport_satisfaction"].mean(),
        #     "Library Satisfaction": df["library_satisfaction"].mean(),
        #     "Local Service Satisfaction": df["local_service_satisfaction"].mean()
        # }
        # st.write(avg_scores)

        st.subheader("AI-Generated Insight Summary")
        sample_text = " ".join(df["local_service_suggestions"].dropna().astype(str).sample(10))
        if sample_text:
            prompt = f"Summarize key citizen feedback themes from the following text: {sample_text}. Also, provide a sentiment score (positive, negative, neutral) for each theme."
            with st.spinner("Generating insights with Gemini..."):
                response = model.generate_content(prompt)
                st.success(response.text)



# --------------------------------------------------------------------------------------------------------------
# Visual Analytics Dashboard
# --------------------------------------------------------------------------------------------------------------
def visual_dashboard():
    df = st.session_state.get("df")
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    st.subheader("Visual Analytics")

    st.sidebar.header("Filters")
    city_filter = st.sidebar.multiselect("Select Cities", options=df["city"].unique(), default=df["city"].unique())
    age_filter = st.sidebar.multiselect("Select Age Groups", options=df["age_group"].unique(), default=df["age_group"].unique())

    filtered_df = df[(df["city"].isin(city_filter)) & (df["age_group"].isin(age_filter))]

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Responses", len(filtered_df))
    col2.metric("Unique Cities", filtered_df["city"].nunique())
    col3.metric("Age Group", filtered_df["age_group"].nunique())
    col4.metric("Avg Serive Use Frequency", filtered_df["service_use_freq"].mode()[0])


    # --- Visualizations ---
    st.subheader("City-wise Service Satisfaction")
    fig1 = px.histogram(filtered_df, x="city", color="local_service_satisfaction",
                        barmode="group", title="Service Satisfaction by City")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Library Satisfaction")
    fig2 = px.pie(filtered_df, names="library_satisfaction", title="Library Satisfaction Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Transport Safety by Gender")
    fig3 = px.histogram(filtered_df, x="gender", color="transport_safety", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📊 Age Group Distribution")
    fig_age = px.histogram(
        filtered_df,
        x='age_group',
        category_orders={"age_group": df['age_group'].value_counts().index.tolist()},
        title="Age Group Distribution",
        color='age_group'
    )
    fig_age.update_layout(xaxis_title="Age Group", yaxis_title="Count", bargap=0.2)
    st.plotly_chart(fig_age, use_container_width=True)

    # 2. Gender distribution
    st.subheader("📊 Gender Distribution")
    fig_gender = px.histogram(
        filtered_df,
        x='gender',
        title="Gender Distribution",
        color= 'gender'
    )
    fig_gender.update_layout(xaxis_title="Gender", yaxis_title="Count", bargap=0.2)
    st.plotly_chart(fig_gender, use_container_width=True)

    # 3. City distribution
    st.subheader("📊 City-wise Feedback Count")
    fig_city = px.histogram(
        filtered_df,
        x='city',
        category_orders={"city": df['city'].value_counts().index.tolist()},
        title="City-wise Feedback Count",
        color_discrete_sequence=["#00CC96"]
    )
    fig_city.update_layout(xaxis_title="City", yaxis_title="Count", bargap=0.2)
    fig_city.update_xaxes(tickangle=45)
    st.plotly_chart(fig_city, use_container_width=True)

    # 4. Service Satisfaction Distribution
    st.subheader("📊 Toilet Service Satisfaction Levels")
    fig_service = px.histogram(
        filtered_df,
        x='toilet_cleanliness',
        title="Local Service Satisfaction Levels",
        color='toilet_cleanliness'
    )
    fig_service.update_layout(xaxis_title="Satisfaction Level", yaxis_title="Count", bargap=0.2)
    st.plotly_chart(fig_service, use_container_width=True)


    # --- WordCloud ---
    st.subheader("WordCloud of Suggestions")
    #text_cols = ["transport_suggestions", "park_suggestions", "library_suggestions", "local_service_suggestions"]
    #all_text = " ".join(filtered_df[col].fillna("") for col in text_cols)
    text = ' '.join(df['local_service_suggestions'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)



# --------------------------------------------------------------------------------------------------------------
# Sentiment SWOT Analysis
# --------------------------------------------------------------------------------------------------------------
def sentiment_swot():
    df = st.session_state.get("df")
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    st.subheader("SWOT Analysis from Feedback")
    if st.button("Generate SWOT Analysis"):
        sample_feedback = " ".join(df["local_service_suggestions"].dropna().astype(str).sample(10))

        prompt = f"Generate a SWOT analysis and Also Genarate a 2x2 table of SWOT:\n\n{sample_feedback}"
        
        with st.spinner("Generating SWOT analysis with Gemini..."):
            response = model.generate_content(prompt)
            st.success(response.text)



# --------------------------------------------------------------------------------------------------------------
# AI Policy Advisor
# --------------------------------------------------------------------------------------------------------------
def ai_policy_advisor():
    st.subheader("AI Policy Advisor for Public Services")
    scenario = st.text_area("Describe a public service scenario:")

    if st.button("Generate Strategies"):
        if scenario:
            prompt = f"Suggest 3 detailed strategies to improve public service based on this scenario: {scenario}"
            with st.spinner("Generating strategies with Gemimi..."):
                response = model.generate_content(prompt)
                st.success(response.text)


# --------------------------------------------------------------------------------------------------------------
# Sentiment Prediction
# --------------------------------------------------------------------------------------------------------------
def predict_sentiment():
    df = st.session_state.get("df")
    if df is None:
        st.warning("Please upload a dataset first.")
        return
    else:
        time.sleep(2)  # Simulate loading time
        st.success("Models loaded successfully! ✅")

        st.subheader("Sentiment Analysis of Citizen Feedback")
        feedback_text = st.text_area("Enter feedback text to analyze sentiment:")
        if st.button("Predict Sentiment"):
            if feedback_text:
                with st.spinner("Analyzing sentiment..."):
                    time.sleep(2)  # Simulate processing time
                    log_sentiment, svm_sentiment = classify_text(feedback_text)

                    if log_sentiment == "Positive" or log_sentiment == "positive":
                        st.success(f"Logistic Regression Sentiment: {log_sentiment}")
                    else:
                        st.error(f"Logistic Regression Sentiment: {log_sentiment}")

                    if svm_sentiment == "Positive" or svm_sentiment == "positive":
                        st.success(f"SVM Sentiment: {svm_sentiment}")
                    else:
                        st.error(f"SVM Sentiment: {svm_sentiment}")
                    
                    st.session_state.feedback_log.append({
                        "text": feedback_text,
                        "logistic_regression": log_sentiment,
                        "svm": svm_sentiment
                    })
            else:
                st.warning("Please enter some feedback text.")



# --------------------------------------------------------------------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------------------------------------------------------------------
menu = st.sidebar.selectbox(
    "Navigate",
    [
        "📂 Upload New Dataset",
        "📈 Citizen Feedback Insights",
        "📊 Visual Analytics Dashboard",
        "🧩 Sentiment SWOT Analysis",
        "🤖 AI Policy Advisor",
        "💭 Predict Sentiment"
    ]
)

if menu == "📂 Upload New Dataset":
    upload_dataset()
elif menu == "📈 Citizen Feedback Insights":
    citizen_feedback_insights()
elif menu == "🤖 AI Policy Advisor":
    ai_policy_advisor()
elif menu == "📊 Visual Analytics Dashboard":
    visual_dashboard()
elif menu == "🧩 Sentiment SWOT Analysis":
    sentiment_swot()
elif menu == "💭 Predict Sentiment":
    predict_sentiment()


st.markdown("©️ 2025 InsightNation. All rights reserved.")