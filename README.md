# InsightNation – Data Analytics Platform for Public Opinion and Smarter Public Service Enhancement

## 🧠 Project Overview

**InsightNation** is a data science-driven platform developed as part of an MBA (Data Science) capstone project. The platform is designed to collect, analyze, and visualize citizen feedback to improve public service delivery and civic engagement. By applying advanced analytics, NLP, and AI techniques (including Google Gemini), the project transforms raw public feedback into actionable insights for administrators, policy makers, and civic planners.

The application processes real-world survey data across multiple domains — such as sanitation, parks, public transport, libraries, and local services — and delivers both statistical and AI-powered interpretations through a user-friendly Streamlit dashboard.

---

## 📂 Folder Structure

```bash
InsightNation/
│
├── data/
│   ├── raw/                # Raw datasets (initial CSV files)
│   ├── processed/          # Cleaned and transformed datasets
│   └── exports/            # Data exports like charts or CSV summaries
│       └── eda_plots/      # Exported EDA visualizations (PNG)
│
├── notebooks/
│   ├── EDA.ipynb           # Exploratory data analysis
│   └── ML_Modeling.ipynb   # ML + NLP pipeline and model training
│
├── src/
│   ├── data_cleaning.py    # Data cleaning and transformation
│   ├── nlp_pipeline.py     # Text preprocessing and sentiment functions
│   ├── model_trainer.py    # Train the model with different machine learning algorithms
│
├── app.py              # Streamlit application file
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and documentation
├── .gitignore              # Git ignore file
└── LICENSE                 # Optional: Project license
```


---

## 🛠️ Technologies & Libraries Used

### 📊 Data Handling & Analysis
- `pandas`
- `numpy`

### 📈 Visualization
- `matplotlib`
- `seaborn`
- `plotly.express`
- `streamlit`

### 🤖 Machine Learning & NLP
- `scikit-learn`
- `spaCy`
- `nltk`
- `wordcloud`
- `re` (regex)
- `joblib` (for saving models)

### 🧠 AI Integration
- `google.generativeai` (Google Gemini API for insights and summarization)

### 📦 Others
- `os`, `pathlib`, `json`, `warnings`, `datetime` (built-in libraries)

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/insightnation.git
   cd 
   ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.
    ```

3. **Run the Streamlit dashboard:**
    ```bash
    streamlit run app.py
    ```

4. **Setup Google Gemini API:**

    Make sure to authenticate and store your Gemini API key as a secure environment variable (e.g., GOOGLE_API_KEY).
    Refer to app.py for Gemini integration section.


## 🌍 Deployment
The final version of the Streamlit app is deployed at:

https://insightnation.streamlit.app

(Hosted via Streamlit Cloud)

## 📌 Key Features
- End-to-end citizen feedback analytics pipeline
- Real-time visualizations with Plotly
- Text summarization and strategic suggestions via Gemini AI
- Predictive modeling using logistic regression and random forests
- NLP-based sentiment classification and keyword extraction
- Interactive Streamlit dashboard
- Modular codebase for scalability and further development

## 📈 Future Enhancements
- Integration with real-time data sources (APIs, social media)
- Support for multilingual NLP processing
- Admin panel for feedback moderation and annotation
- Role-based access control for different civic departments
- Automated monthly reports with AI-driven summaries

## 👨‍🎓 Author
Pranoy Chakraborty

MBA (Data Science) – Amity University Online, 2023–25

Final Capstone Project – Semester 4

