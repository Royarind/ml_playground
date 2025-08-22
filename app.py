# Created by @Arindam Roy 08-2025

from __future__ import annotations
import io
import glob
import zipfile
import json
import re
import os
import time
import platform
from typing import Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from sklearn.base import clone


import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import optuna
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#Models

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.base import is_classifier
import xgboost as xgb
import lightgbm as lgb

# Groq LLM integration
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Optional imports with try-except
try:
    import pkg_resources
except Exception:
    pkg_resources = None

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    RandomizedSearchCV, 
    cross_val_score
)
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler,
    OneHotEncoder, 
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import (
    LogisticRegression, 
    LinearRegression, 
    Ridge, 
    Lasso
)
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor, 
    GradientBoostingClassifier,
    IsolationForest
)
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    confusion_matrix, 
    ConfusionMatrixDisplay,
    r2_score, 
    mean_absolute_error, 
    mean_squared_error,
    roc_curve, 
    precision_recall_curve, 
    classification_report,
    make_scorer
)

from optuna.visualization import plot_optimization_history, plot_param_importances

# Optional deps
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer, WordNetLemmatizer
except Exception:  # pragma: no cover
    nltk = None
    stopwords = None
    SnowballStemmer = None
    WordNetLemmatizer = None

try:
    import statsmodels.api as sm  # enables trendline='ols' in px.scatter
except Exception:  # pragma: no cover
    sm = None

try:
    from ydata_profiling import ProfileReport
except Exception:  # pragma: no cover
    ProfileReport = None

try:
    import streamlit.components.v1 as components
except Exception:  # pragma: no cover
    components = None

# Optional imports guarded to avoid runtime errors if not installed
try:
    import seaborn as sns  # for sample datasets
except Exception:  # pragma: no cover
    sns = None

try:
    from sklearn import datasets as skdatasets
except Exception:  # pragma: no cover
    skdatasets = None

# Run ONCE at the very top
st.set_page_config(page_title="ML Playground", layout="wide", page_icon="")

#load_dotenv()

# Initialize Groq client
def init_groq_client():
    """Initialize Groq client with API key from environment or secrets"""
    if not GROQ_AVAILABLE:
        st.sidebar.warning("Groq library not available")
        return None
    
    # Try to get API key from environment variable (production)
    api_key = os.environ.get('GROQ_API_KEY')
    
    # If not found, try Streamlit secrets (development)
    if not api_key:
        try:
            api_key = st.secrets.get('GROQ_API_KEY', '')
        except:
            api_key = ''
    
    # If still not found, show warning but don't crash
    # if not api_key:
    #     st.sidebar.warning("â ï¸ Groq API key not configured")
    #     st.session_state.groq_available = False
    #     return None
    
    # Initialize client
    if 'groq_client' not in st.session_state:
        try:
            st.session_state.groq_client = Groq(api_key=api_key)
            st.session_state.groq_available = True
            #st.sidebar.success("ð¤ Groq LLM Available")
        except Exception as e:
            st.error(f"Failed to initialize Groq client: {e}")
            st.session_state.groq_available = False
    
    return st.session_state.get('groq_client')
def init_groq_client():
    """Initialize Groq client with API key from environment or secrets"""
    if not GROQ_AVAILABLE:
        st.sidebar.warning("Groq library not available")
        return None
    
    # Try to get API key from environment variable (production)
    api_key = os.environ.get('GROQ_API_KEY')
    
    # If not found, try Streamlit secrets (development)
    if not api_key:
        try:
            api_key = st.secrets.get('GROQ_API_KEY', '')
        except:
            api_key = ''
    
    # If still not found, show warning but don't crash
    if not api_key:
        st.sidebar.warning("Groq API key not configured")
        st.session_state.groq_available = False
        return None
    
    # Initialize client
    if 'groq_client' not in st.session_state:
        try:
            st.session_state.groq_client = Groq(api_key=api_key)
            st.session_state.groq_available = True
            #st.sidebar.success("ð¤ Groq LLM Available")
        except Exception as e:
            st.error(f"Failed to initialize Groq client: {e}")
            st.session_state.groq_available = False
    
    return st.session_state.get('groq_client')
def call_groq_llm(prompt, model="llama3-70b-8192", max_tokens=1024, temperature=0.7, timeout=30):
    """Call Groq LLM API with proper error handling and timeouts"""
    if not GROQ_AVAILABLE:
        return "Groq API not available. Please configure your API key."
    
    try:
        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stream=False,
            stop=None,
            timeout=timeout  # Add timeout
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        # More specific error handling
        error_msg = str(e).lower()
        
        if "connection" in error_msg or "timeout" in error_msg:
            return "Connection error: Unable to reach Groq servers. Please check your internet connection and try again."
        elif "authentication" in error_msg or "invalid" in error_msg or "401" in error_msg:
            return "Authentication error: Invalid API key. Please check your Groq API key."
        elif "rate" in error_msg or "429" in error_msg:
            return "Rate limit exceeded: Please wait a moment and try again."
        elif "permission" in error_msg or "403" in error_msg:
            return "Permission denied: Your API key may not have access to this model."
        else:
            return f"Error calling Groq API: {str(e)}"

def groq_data_analysis(prompt_suffix, df, context=""):
    """Helper function to analyze data with Groq LLM"""
    if df is None:
        return "No dataset available for analysis."
    
    base_prompt = f"""
    You are a data scientist assistant. Analyze this dataset and provide helpful insights.
    
    Dataset shape: {df.shape}
    Columns: {', '.join(df.columns.tolist())}
    Sample data (first 3 rows):
    {df.head(3).to_string()}
    
    Data types:
    {df.dtypes.to_string()}
    
    {context}
    
    {prompt_suffix}
    
    Please provide a concise, helpful response focused on practical insights.
    """
    
    return call_groq_llm(base_prompt)

# Sidebar navigation
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Data Loading",
        "EDA",
        "Train-Test Split",
        "Pipeline and Model Training",  # Replace "Pipeline" and "Training" with this
        "Final Evaluation",
        "Export",
        "Prediction",
    ]
)

#Session State

def ensure_session_state():
    """Initialize expected keys in st.session_state."""
    ss = st.session_state
    ss.setdefault("df", None)                
    ss.setdefault("versions", [])             
    ss.setdefault("dataset_name", None)       
    ss.setdefault("notices", [])             
    ss.setdefault("groq_available", GROQ_AVAILABLE)


def inject_css():
    """Apply consistent minimal styling across the app (buttons, cards, spacing)."""
    st.markdown(
        """
        <style>
        /* Primary button styling */
        div[data-testid="stButton"] button {
            background-color: #0a66c2; /* LinkedIn blue */
            color: white; border: none; border-radius: 10px;
            padding: 0.6em 1.0em; font-weight: 600; cursor: pointer;
            transition: background-color .2s ease;
        }
        div[data-testid="stButton"] button:hover { background-color: #0b5cad; }
        /* Secondary buttons */
        .secondary-btn button { background-color: #eef3f8 !important; color: #0a66c2 !important; }
        /* Cards */
        .card { padding: 1rem; border: 1px solid #e6e6e6; border-radius: 14px; background: #ffffff; }
        .muted { color: #6b7280; }
        .warn { color: #a16207; }
        .success { color: #046c4e; }
        /* Reduce empty space slightly */
        .block-container { padding-top: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _read_dataframe_from_upload(file) -> pd.DataFrame:
    """Read CSV/Excel based on file name; robust to common encodings."""
    name = file.name.lower()
    if name.endswith((".csv", ".txt")):
        
        try:
            return pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin-1")
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def load_sample_dataset(sample_key: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Return a (df, name) pair for a chosen sample dataset from seaborn/sklearn."""
    if not sample_key:
        return None, None

    # Seaborn datasets
    if sample_key.startswith("sns:") and sns is not None:
        name = sample_key.split(":", 1)[1]
        try:
            df = sns.load_dataset(name)
            return df, f"seaborn_{name}"
        except Exception:
            st.warning("Could not load seaborn dataset. Is seaborn available?")
            return None, None

    # Scikit-learn toy datasets
    if sample_key.startswith("sk:") and skdatasets is not None:
        name = sample_key.split(":", 1)[1]
        try:
            if name == "iris":
                data = skdatasets.load_iris(as_frame=True)
            elif name == "wine":
                data = skdatasets.load_wine(as_frame=True)
            elif name == "breast_cancer":
                data = skdatasets.load_breast_cancer(as_frame=True)
            elif name == "diabetes":
                data = skdatasets.load_diabetes(as_frame=True)
            elif name == "digits":
                data = skdatasets.load_digits(as_frame=True)
            else:
                st.warning("Unknown scikit-learn dataset key.")
                return None, None
            df = data.frame
            return df, f"sk_{name}"
        except Exception:
            st.warning("Could not load scikit-learn dataset.")
            return None, None

    st.info("Select a valid sample dataset.")
    return None, None


def upload_panel(key_prefix: str = "home") -> Optional[pd.DataFrame]:
    
    st.markdown("### Upload Data")

    tabs = st.tabs(["File", "Reminder"])
    with tabs[0]:
        uploaded_files = st.file_uploader(
            "Upload CSV/Excel (single or multiple) — we will use the first file here; advanced merge is on the EDA page",
            type=["csv", "txt", "xls", "xlsx"],
            accept_multiple_files=True,
            key=f"{key_prefix}_uploader",
        )
        if uploaded_files:
            file = uploaded_files[0]
            try:
                df = _read_dataframe_from_upload(file)
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                return None
            st.success(f"Loaded: {file.name}")
            return df


    with tabs[1]:
        st.markdown(
            "- You can always **download the current dataset** (CSV/XLSX) using the panel below.\n"
            "- **Do not rename** the downloaded file if you plan to reload it later to continue work.\n"
            "- Advanced sources (URL, Google Sheets, SQL, Kaggle) are available on the **Data Loading** page.")
    return None


def download_panel(df: Optional[pd.DataFrame], filename_basename: str = "dataset"):
    """Provide CSV and Excel download buttons for the given df.
    If df is None, render disabled controls with guidance.
    """
    st.markdown("### Download Current Data")
    if df is None:
        st.info("No active dataset. Upload or load a sample above.")
        return

    # CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"{filename_basename}.csv",
        mime="text/csv",
        key="dl_csv",
    )

    # Excel
    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    st.download_button(
        label="Download Excel",
        data=xls_buf.getvalue(),
        file_name=f"{filename_basename}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_xlsx",
    )


def preview_card(df: Optional[pd.DataFrame], name: Optional[str] = None):
    """Show a compact preview with shape and dtypes."""
    st.markdown("### Dataset Overview")
    if df is None:
        st.info("No dataset loaded yet.")
        return
    r, c = df.shape
    st.markdown(
        f"<div class='card'>"
        f"<b>Name:</b> {name or 'current'} &nbsp; • &nbsp; <b>Rows:</b> {r} &nbsp; • &nbsp; <b>Columns:</b> {c}"
        f"</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Peek first 10 rows"):
        st.dataframe(df.head(10), use_container_width=True)
    with st.expander("Column dtypes"):
        st.write(df.dtypes)

##############################################################
# Page: 0 Landing page for the ML Playground. Has all the instructions
##############################################################


if page == "Home":
    st.title("ML Playground - Your AI-Powered Machine Learning Workbench")
    
    ensure_session_state()
    inject_css()
    
    # Initialize Groq with better status display
    groq_client = init_groq_client()
    
    # AI Status Banner
    if st.session_state.groq_available:
        st.success("AI Assistant Enabled - Get smart recommendations throughout your workflow!")
    else:
        st.warning("AI Assistant Not Configured - Add GROQ_API_KEY to secrets.toml for AI-powered guidance")
    
    # Quick Start Section
    st.markdown("## Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Load Sample Data", key="quick_sample"):
            st.session_state.df = sns.load_dataset("iris")
            st.session_state.dataset_name = "iris_sample"
            st.success("Iris dataset loaded! Go to EDA to explore.")
    
    with col2:
        if st.button("🔄 Restart Session", key="quick_restart"):
            keys_to_keep = ['groq_available', 'groq_client']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.success("Session reset! Ready for a new project.")
    
    with col3:
        if st.button("📚 View Tutorial", key="quick_tutorial"):
            st.info("Check each step's expander for detailed instructions and AI tips!")

    # Step-by-step guide with AI enhancements
    st.markdown("## 📋 Step-by-Step Workflow")
    
    # Step 1: Data Loading
    with st.expander("✅ Step 1: Load Your Data", expanded=False):
        st.markdown("""
        **Start by loading your dataset - AI can help you choose the right one!**
        
        ### 📁 Data Sources:
        - **Upload File**: CSV/Excel files from your computer
        - **Preloaded Datasets**: 20+ curated datasets (Iris, Wine, Titanic, Diabetes, Breast Cancer, California Housing, Penguins, etc.)
        - **From URL**: Load from public URLs
        - **SQL Database**: Connect to your database
        - **Google Sheets**: Import from Google Sheets
        - **Kaggle**: Download datasets from Kaggle

        ### ✅ AI Assistance Available:
        - **Dataset Recommendations**: AI suggests datasets based on your problem type
        - **Data Quality Check**: Automatic analysis of uploaded data
        - **Problem Type Detection**: AI helps identify classification vs regression

        **→ Action: Go to →** **Data Loading** page in the sidebar
        
        💡 **Pro Tip**: Start with `Iris` for classification or `Diabetes` for regression to learn the workflow!
        """)
        
        if st.session_state.groq_available:
            if st.button("✅ Get Dataset Recommendation", key="ai_dataset_rec"):
                response = call_groq_llm(
                    "Suggest 3 good starter datasets for machine learning beginners, "
                    "including one classification, one regression, and one real-world dataset. "
                    "Explain why each is good for learning."
                )
                st.info(response)

    # Step 2: EDA
    with st.expander("🔍 Step 2: EDA - Explore & Understand Your Data"):
        st.markdown("""
        **Explore, clean, and understand your data with AI-powered insights**
        
        ### 🛠️ Tools Available:
        - **Data Overview**: Summary statistics, missing values, data types
        - **Data Cleaning**: Handle missing values, outliers, duplicates with smart suggestions
        - **Feature Engineering**: Create new features, transform variables
        - **Interactive Visualizations**: Charts, plots, and correlation analysis
        - **Automated Profiling**: Comprehensive data profile reports
        - **Text Processing**: Clean and preprocess text columns

        ### ✅ AI Assistance Available:
        - **Automatic Insights**: AI identifies patterns and issues in your data
        - **Cleaning Recommendations**: Smart suggestions for handling missing data
        - **Feature Engineering Ideas**: AI suggests new features to create
        - **Visualization Recommendations**: Best charts for your data types

        **→ Action: Go to →** **EDA** page in the sidebar
        
        💡 **Pro Tip**: Use the snapshot feature to compare before/after changes and ask AI for cleaning strategies!
        """)

    # Step 3: Train-Test Split
    with st.expander("✂️ Step 3: Split Your Data for Training"):
        st.markdown("""
        **Prepare your data for model training with optimal splitting strategies**
        
        ### ⚙️ Configuration:
        - **Select Target Column**: Choose what you want to predict (AI can help identify the best target)
        - **Split Parameters**: Set test size, random state, validation split
        - **Stratification**: Automatically handle imbalanced classes
        - **Download Splits**: Export train/validation/test sets

        ### ✅ AI Assistance Available:
        - **Target Selection Help**: AI suggests the most appropriate target variable
        - **Optimal Split Ratios**: AI recommends best train/test/validation splits for your data size
        - **Stratification Advice**: Guidance on when to use stratified sampling

        **→ Action: Go to →** **Train-Test Split** page in the sidebar
        
        💡 **Pro Tip**: For small datasets (<1000 samples), use 80/20 split. For larger datasets, 70/15/15 works well!
        """)

    # Step 4: Pipeline Building
    with st.expander("⚙️ Step 4: Build Your ML Pipeline"):
        st.markdown("""
        **Create smart preprocessing and modeling pipelines with AI guidance**
        
        ### 🏗️ Pipeline Components:
        - **Column Assignment**: Specify numeric vs categorical features (AI can auto-detect)
        - **Numeric Pipeline**: Imputation, scaling strategies with smart defaults
        - **Categorical Pipeline**: Encoding, imputation methods with best practices
        - **Model Selection**: Choose from 10+ algorithms for your problem type
        - **Auto-Training**: Train multiple models and compare performance

        ### ✅ AI Assistance Available:
        - **Pipeline Recommendations**: AI suggests optimal preprocessing steps
        - **Model Selection Guide**: AI recommends best algorithms for your data type
        - **Hyperparameter Starting Points**: Smart default values for each model
        - **Baseline Comparison**: AI helps interpret baseline model performance

        **→ Action: Go to →** **Pipeline and Model Training** page in the sidebar
        
        💡 **Pro Tip**: Start with Logistic Regression/Linear Regression as baselines before trying complex models!
        """)

    # Step 5: Training & Tuning
    with st.expander("🎯Step 5: Train & Optimize Models"):
        st.markdown("""
        **Fine-tune your models with advanced optimization techniques**
        
        ### 🔧 Optimization Methods:
        - **Hyperparameter Tuning**: Grid search, random search, and Optuna optimization
        - **Cross-Validation**: K-fold validation for robust evaluation
        - **Early Stopping**: Prevent overfitting with smart stopping criteria
        - **Performance Tracking**: Monitor training progress and metrics

        ### ✅ AI Assistance Available:
        - **Tuning Strategy Advice**: AI recommends best search method for your problem
        - **Parameter Space Guidance**: Smart parameter ranges based on your data
        - **Overfitting Detection**: AI helps identify when models are overfitting
        - **Optimization Tips**: Recommendations for faster convergence

        **→ Action: Go to →** **Pipeline and Model Training** page in the sidebar
        
        💡 **Pro Tip**: Use Random Search for quick results, Grid Search for exhaustive tuning, and Optuna for smart optimization!
        """)

    # Step 6: Evaluation
    with st.expander("📊 Step 6: Evaluate Model Performance"):
        st.markdown("""
        **Comprehensively evaluate your models with detailed metrics and explanations**
        
        ### 📈 Evaluation Tools:
        - **Performance Metrics**: Accuracy, Precision, Recall, F1, R², MAE, RMSE, etc.
        - **Confusion Matrix**: Visualize classification performance
        - **ROC & Precision-Recall Curves**: Analyze model discrimination ability
        - **SHAP Analysis**: Understand feature importance and model decisions
        - **Model Comparison**: Compare multiple models side-by-side

        ### ✅ AI Assistance Available:
        - **Metric Interpretation**: AI explains what each metric means for your problem
        - **Error Analysis**: AI helps identify patterns in model mistakes
        - **Feature Importance Explanation**: AI interprets SHAP results in plain English
        - **Model Selection Help**: AI recommends which model to choose for deployment

        **→ Action: Go to →** **Final Evaluation** page in the sidebar
        
        💡 **Pro Tip**: Don't just look at accuracy! Consider precision/recall tradeoffs for classification and multiple error metrics for regression!
        """)

    # Step 7: Export & Deployment
    with st.expander("📦 Step 7: Export & Deploy Your Model"):
        st.markdown("""
        **Package your complete solution for deployment and sharing**
        
        ### 🚀 Export Options:
        - **Model Export**: Save trained models and complete pipelines
        - **Project Bundles**: Package everything into a single ZIP file
        - **Metadata & Documentation**: Automatic project documentation
        - **Requirements File**: Generate dependency list for reproducibility
        - **Multiple Formats**: Export as joblib, pickle, or ONNX

        ### ✅ AI Assistance Available:
        - **Deployment Advice**: AI suggests best deployment options for your model type
        - **Documentation Generation**: AI helps create model cards and documentation
        - **Versioning Recommendations**: Smart version numbering and change tracking
        - **Reproducibility Checks**: AI verifies your bundle contains everything needed

        **→ Action: Go to →** **Export** page in the sidebar
        
        💡 **Pro Tip**: Always include a README with your exported models explaining the problem, data, and model performance!
        """)

    # Step 8: Prediction
    with st.expander("🔮 Step 8: Make Predictions with Your Model"):
        st.markdown("""
        **Use your trained models for real-world predictions and inference**
        
        ### 🎯 Prediction Modes:
        - **Single Prediction**: Input values for one prediction at a time
        - **Batch Prediction**: Upload CSV/Excel files for multiple predictions
        - **Model Comparison**: Test multiple models on the same data
        - **Probability Scores**: See prediction confidence levels and uncertainty
        - **Class Decoding**: Get human-readable predictions with explanations

        ### ✅ AI Assistance Available:
        - **Prediction Interpretation**: AI explains what the predictions mean
        - **Uncertainty Analysis**: AI helps interpret probability scores and confidence
        - **Error Checking**: AI validates input data before prediction
        - **Result Explanation**: AI provides context for prediction results

        **→ Action: Go to →** **Prediction** page in the sidebar
        
        💡 **Pro Tip**: Test your model with edge cases and ask AI to explain why it made certain predictions!
        """)

    # Interactive AI Assistant Section
    st.markdown("---")
    st.markdown("## AI Assistant Chat")
    
    if st.session_state.groq_available:
        chat_col1, chat_col2 = st.columns([3, 1])
        
        with chat_col1:
            user_question = st.text_area(
                "Ask the AI Assistant anything about ML:",
                placeholder="e.g., 'Which model should I use for my data?', 'How do I handle missing values?', 'Explain overfitting to me'...",
                height=100,
                key="ai_chat_input"
            )
        
        with chat_col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("Ask AI Assistant", key="ai_chat_ask"):
                if user_question.strip():
                    with st.spinner("Thinking..."):
                        response = call_groq_llm(
                            f"You are an expert ML consultant. Help the user with their machine learning question: {user_question}"
                            "\n\nProvide clear, practical advice suitable for beginners to intermediate users."
                            "\nInclude examples when helpful and explain concepts simply."
                        )
                        st.success("💡 AI Response:")
                        st.info(response)
                else:
                    st.warning("Please enter a question first.")
    else:
        st.info("💡 Enable AI Assistant by adding your GROQ_API_KEY to secrets.toml for personalized guidance")

    # Quick status dashboard
    st.markdown("---")
    st.markdown("## Current Project Status")
    
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        if st.session_state.df is not None:
            st.success("Data Loaded")
            st.caption(f"{st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
        else:
            st.warning("No Data")
            st.caption("Complete Step 1")
    
    with status_col2:
        if "split_result" in st.session_state:
            st.success("Data Split")
            split = st.session_state["split_result"]
            train_size = len(split["X_train"])
            test_size = len(split["X_test"])
            st.caption(f"Train: {train_size}, Test: {test_size}")
        else:
            st.info("Ready for Split")
            st.caption("Complete Step 3")
    
    with status_col3:
        if 'trained_models' in st.session_state:
            st.success("Models Trained")
            st.caption(f"{len(st.session_state.trained_models)} model(s) trained")
        else:
            st.info("Ready for Training")
            st.caption("Complete Step 4-5")
    
    with status_col4:
        if 'best_model' in st.session_state:
            st.success("Best Model Ready")
            best_name = st.session_state.best_model['name'].replace('BEST_', '')
            st.caption(f"{best_name} selected")
        else:
            st.info("Best Model Pending")
            st.caption("Complete Step 6")

    # Quick actions
    st.markdown("---")
    st.markdown("## Quick Actions")
    
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.session_state.df is not None:
            st.markdown("### Data Preview")
            st.dataframe(st.session_state.df.head(3))
            st.caption(f"Target: {st.session_state.get('target_column', 'Not set')}")
    
    with quick_col2:
        if st.session_state.df is not None:
            st.markdown("### 💾 Download Options")
            download_panel(st.session_state.df, 
                         filename_basename=st.session_state.get("dataset_name") or "dataset")
    
    with quick_col3:
        st.markdown("### Need Help?")
        st.markdown("""
        - Check each step's instructions above
        - Use the AI Assistant chat for questions
        - Hover over options for tooltips
        - Save your work frequently with exports
        
        **Keyboard Shortcuts:**
        - `r` - Refresh current page
        - `s` - Save current state
        - `a` - Open AI chat
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>ML Playground v2.0 | AI-Powered Workflow</p>
        <p>Start with Step 1 above or use the sidebar to jump to any step!</p>
    </div>
    """, unsafe_allow_html=True)


################################################
# Page 1: Data Loading
################################################

elif page == "Data Loading":
    st.header("Data Loading")
    st.markdown("# Data Loading")
    st.caption("Load your dataset from multiple sources — file, preloaded, URL, SQL, Google Sheets, or Kaggle")

    ensure_session_state()
    inject_css()
    init_groq_client()

    # Two-column layout
    left, right = st.columns([1, 1])

    with left:
        st.markdown("## Load Dataset")

        source = st.radio(
            "Select Data Source",
            ["Upload File", "Preloaded Dataset", "From URL", "SQL Database", "Google Sheets", "Kaggle Dataset"]
        )

        # --- Option 1: Upload File ---
        if source == "Upload File":
            df = upload_panel(key_prefix="data_page")
            if df is not None:
                st.session_state.df = df
                st.session_state.dataset_name = "uploaded_file"
                st.session_state.versions.append(("upload", df.copy()))
                st.success(f"Dataset loaded! Shape: {df.shape}")

        # --- Option 2: Preloaded Dataset (sns/sklearn) ---
        # Update your Data Loading page - replace the preloaded dataset section with this:

# --- Option 2: Preloaded Dataset (sns/sklearn) ---
        elif source == "Preloaded Dataset":
            preload_choice = st.selectbox(
                "Choose dataset",
                [
                    "(choose)",
                    # Scikit-learn datasets
                    "sk:iris", "sk:wine", "sk:breast_cancer", "sk:diabetes", "sk:digits",
                    "sk:linnerud", "sk:california_housing", "sk:olivetti_faces",
                    
                    # Seaborn datasets  
                    "sns:iris", "sns:tips", "sns:titanic", "sns:planets", "sns:fmri",
                    "sns:dots", "sns:flights", "sns:mpg", "sns:geyser", "sns:penguins",
                    "sns:car_crashes", "sns:anagrams", "sns:attention", "sns:exercise",
                    
                    # Classification-focused
                    "classification", "regression", "mixed"
                ]
            )
            
            if st.button("Load Preloaded Dataset", key="load_preloaded") and preload_choice != "(choose)":
                df, name = load_sample_dataset(preload_choice)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.dataset_name = name
                    st.session_state.versions.append(("preloaded", df.copy()))
                    st.success(f"Loaded {name}! Shape: {df.shape}")

        # --- Option 3: From URL ---
        elif source == "From URL":
            url = st.text_input("Enter URL to CSV or Excel file")
            if st.button("Load from URL", key= "Load Data") and url:
                try:
                    if url.lower().endswith(".csv"):
                        df = pd.read_csv(url)
                    elif url.lower().endswith((".xls", ".xlsx")):
                        df = pd.read_excel(url)
                    else:
                        raise ValueError("Unsupported format. Must be CSV/XLSX.")
                    st.session_state.df = df
                    st.session_state.dataset_name = "url_data"
                    st.session_state.versions.append(("url", df.copy()))
                    st.success(f"Dataset loaded from URL! Shape: {df.shape}")
                except Exception as e:
                    st.error(f"Error loading from URL: {e}")

        # --- Option 4: SQL Database ---
        elif source == "SQL Database":
            db_uri = st.text_input("Enter database URI", help="Example: sqlite:///mydb.sqlite")
            query = st.text_area("Enter SQL query", help="Example: SELECT * FROM mytable")
            if st.button("Load from SQL", key= "Load from SQL") and db_uri and query:
                try:
                    from sqlalchemy import create_engine
                    engine = create_engine(db_uri)
                    df = pd.read_sql(query, con=engine)
                    st.session_state.df = df
                    st.session_state.dataset_name = "sql_data"
                    st.session_state.versions.append(("sql", df.copy()))
                    st.success(f"Dataset loaded from SQL! Shape: {df.shape}")
                except Exception as e:
                    st.error(f"Error loading from SQL: {e}")

        # --- Option 5: Google Sheets ---
        elif source == "Google Sheets":
            sheet_url = st.text_input("Enter Google Sheets URL")
            if st.button("Load from Google Sheets", key= "Load from Gsheet") and sheet_url:
                try:
                    if "docs.google.com" in sheet_url:
                        sheet_id = sheet_url.split("/")[5]
                        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                        df = pd.read_csv(csv_url)
                        st.session_state.df = df
                        st.session_state.dataset_name = "google_sheets"
                        st.session_state.versions.append(("gsheet", df.copy()))
                        st.success(f"Dataset loaded from Google Sheets! Shape: {df.shape}")
                    else:
                        st.error("Invalid Google Sheets link.")
                except Exception as e:
                    st.error(f"Error loading Google Sheets: {e}")

        # --- Option 6: Kaggle Dataset ---
        elif source == "Kaggle Dataset":
            st.info("Upload your Kaggle API credentials (kaggle.json) to connect.")
            kaggle_file = st.file_uploader("Upload kaggle.json", type=["json"])
            dataset_id = st.text_input("Enter Kaggle dataset identifier", help="e.g., zynicide/wine-reviews")

            if kaggle_file and dataset_id and st.button("Download & Load from Kaggle", "Kaggle"):
                try:
                    import json, os, kaggle, tempfile, glob
                    tmp_dir = tempfile.mkdtemp()
                    with open(os.path.join(tmp_dir, "kaggle.json"), "wb") as f:
                        f.write(kaggle_file.getbuffer())
                    os.environ["KAGGLE_CONFIG_DIR"] = tmp_dir

                    kaggle.api.authenticate()
                    kaggle.api.dataset_download_files(dataset_id, path=tmp_dir, unzip=True)

                    csv_files = glob.glob(os.path.join(tmp_dir, "**", "*.csv"), recursive=True)
                    if csv_files:
                        df = pd.read_csv(csv_files[0])
                        st.session_state.df = df
                        st.session_state.dataset_name = "kaggle_data"
                        st.session_state.versions.append(("kaggle", df.copy()))
                        st.success(f"Dataset loaded from Kaggle! Shape: {df.shape}")
                    else:
                        st.error("No CSV found in Kaggle dataset.")
                except Exception as e:
                    st.error(f"Error loading from Kaggle: {e}")

    # --- Preview & Download Panel ---
    with right:
        preview_card(st.session_state.df, st.session_state.get("dataset_name"))
        
        # AI Data Analysis
        if st.session_state.df is not None and st.session_state.groq_available:
            with st.expander("🤖 AI Data Analysis"):
                if st.button("Analyze Dataset with AI", key="AI"):
                    with st.spinner("Analyzing your data..."):
                        analysis = groq_data_analysis(
                            "Provide a comprehensive analysis of this dataset including:\n"
                            "1. Data quality assessment\n"
                            "2. Potential issues or anomalies\n"
                            "3. Suggested preprocessing steps\n"
                            "4. Recommended ML approaches based on the data characteristics",
                            st.session_state.df
                        )
                        st.info(analysis)

    st.markdown("---")
    download_panel(st.session_state.df, filename_basename=st.session_state.get("dataset_name") or "dataset")
    #st.warning("**Reminder:** Save your progress by downloading the dataset before closing. Do not rename the file if you plan to reload it later.")


################################################
# Page 2: EDA
################################################

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    ensure_session_state()
    inject_css()
    init_groq_client()

    st.markdown("# Exploratory Data Analysis")
    st.caption("Explore your dataset: summary, cleaning, outliers, text, datetime, visualizations, and full profiling.")

    if st.session_state.df is None:
        st.warning("No dataset loaded. Please upload on Home / Data Loading page.")
        st.stop()

    # Handy locals
    DF_KEY = "df"
    df: pd.DataFrame = st.session_state[DF_KEY]

    # A snapshot to enable Before vs After comparisons
    st.session_state.setdefault("eda_snapshot", None)

    # ---------- helpers ----------

    def snapshot_current(label: str = "snapshot"):
        st.session_state.eda_snapshot = (label, st.session_state[DF_KEY].copy())
        st.success(f"Saved snapshot: {label}")

    @st.cache_data(show_spinner=False)
    def _value_counts_safe(s: pd.Series, dropna: bool = False):
        try:
            return s.value_counts(dropna=dropna)
        except Exception:
            return pd.Series(dtype="int64")

    # Generic chart builder so we can re-use for Before/After

    def make_chart(df_: pd.DataFrame,
                kind: str,
                x: str | None,
                y: str | None,
                z: str | None = None,
                color: str | None = None,
                size: str | None = None,
                symbol: str | None = None,
                facet_row: str | None = None,
                facet_col: str | None = None,
                marginal_x: str | None = None,
                marginal_y: str | None = None,
                trendline: str | None = None,
                animation_frame: str | None = None,
                template: str = "plotly",
                title: str | None = None):
        if kind == "Histogram":
            fig = px.histogram(df_, x=x, color=color, marginal=marginal_y, template=template)
        elif kind == "Box":
            fig = px.box(df_, x=x, y=y, color=color, template=template)
        elif kind == "Violin":
            fig = px.violin(df_, x=x, y=y, color=color, box=True, points=False, template=template)
        elif kind == "Scatter":
            fig = px.scatter(df_, x=x, y=y, color=color, size=size, symbol=symbol,
                            facet_row=facet_row, facet_col=facet_col,
                            marginal_x=marginal_x, marginal_y=marginal_y,
                            trendline=trendline if trendline in ("ols", "lowess") else None,
                            template=template, animation_frame=animation_frame)
        elif kind == "Line":
            fig = px.line(df_, x=x, y=y, color=color, template=template)
        elif kind == "Area":
            fig = px.area(df_, x=x, y=y, color=color, template=template)
        elif kind == "Pie":
            fig = px.pie(df_, names=x, values=y, template=template)
        elif kind == "Density 2D":
            fig = px.density_contour(df_, x=x, y=y, template=template)
        elif kind == "Heatmap":
            fig = px.density_heatmap(df_, x=x, y=y, template=template)
        elif kind == "3D Scatter":
            fig = px.scatter_3d(df_, x=x, y=y, z=z, color=color, size=size, template=template)
        elif kind == "3D Surface":
            # requires pivotable grid
            if x and y and z and df_[x].nunique()*df_[y].nunique() <= 20_000:
                zz = df_.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
                fig = go.Figure(data=[go.Surface(z=zz.values, x=zz.columns, y=zz.index)])
            else:
                fig = go.Figure()
        elif kind == "Animated Scatter":
            fig = px.scatter(df_, x=x, y=y, color=color, size=size, animation_frame=animation_frame, template=template)
        elif kind == "Animated Bar":
            fig = px.bar(df_, x=x, y=y, color=color, animation_frame=animation_frame, template=template)
        else:
            fig = px.scatter(df_, x=x, y=y, template=template)

        if title:
            fig.update_layout(title=title)
        fig.update_layout(legend_title_text=color or "Legend")
        return fig
    
    # ---------- AI analysis ----------
    if st.session_state.groq_available:
        with st.expander("✅ AI-Powered Data Analysis", expanded=False):
            analysis_type = st.selectbox("Analysis Type", 
                                       ["General Overview", "Data Quality", "Feature Suggestions", 
                                        "ML Readiness", "Custom Question"])
            
            custom_question = ""
            if analysis_type == "Custom Question":
                custom_question = st.text_input("Ask a specific question about your data")
            
            if st.button("Run AI Analysis", key="AI 2"):
                with st.spinner("Analyzing data with AI..."):
                    if analysis_type == "General Overview":
                        prompt = "Provide a comprehensive overview of this dataset, including key characteristics, patterns, and potential insights."
                    elif analysis_type == "Data Quality":
                        prompt = "Analyze data quality issues including missing values, outliers, data type problems, and suggest cleaning steps."
                    elif analysis_type == "Feature Suggestions":
                        prompt = "Suggest new features that could be engineered from this data and explain their potential value."
                    elif analysis_type == "ML Readiness":
                        prompt = "Assess this dataset's readiness for machine learning and suggest appropriate algorithms and preprocessing steps."
                    else:
                        prompt = custom_question
                    
                    analysis = groq_data_analysis(prompt, df)
                    st.info(analysis)

    # ---------- top summary ----------

    with st.expander("Dataset Overview", expanded=False):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
        with c3:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            st.metric("Numeric / Categorical", f"{len(num_cols)} / {len(cat_cols)}")

        st.markdown("---")
        sub1, sub2 = st.columns([1,1])
        with sub1:
            st.markdown("**Missing values by column**")
            st.dataframe(df.isna().sum().to_frame("missing"))
        with sub2:
            st.markdown("**Sample (head)**")
            st.dataframe(df.head(10), use_container_width=True)

    with st.expander(" Describe (numeric & categorical)"):
        st.dataframe(df.describe(include="all").T)

        st.button("Snapshot current dataset for Before/After", on_click=snapshot_current, kwargs={"label": "overview"}, key= "1")

        

    # ---------- combine/append/merge ----------
    with st.expander("Combine Datasets (append / merge)", expanded=False):
        tabs = st.tabs(["Append (rows)", "Merge (SQL-style)"])
        with tabs[0]:
            files = st.file_uploader("Upload CSV/Excel to append (multiple allowed)", type=["csv","xlsx"], accept_multiple_files=True)
            if files and st.button("Append to current",key="2"):
                new_parts = []
                for f in files:
                    try:
                        new_parts.append(pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f))
                    except Exception as e:
                        st.error(f"Failed {f.name}: {e}")
                if new_parts:
                    st.session_state[DF_KEY] = pd.concat([st.session_state[DF_KEY]] + new_parts, ignore_index=True)
                    st.success("Appended data.")
                    st.success("Appended data.")
                    st.rerun()

        with tabs[1]:
            st.info("Merge with another dataset using SQL-style joins")
            merge_file = st.file_uploader("Upload dataset to merge", type=["csv","xlsx"])
            if merge_file:
                try:
                    merge_df = pd.read_csv(merge_file) if merge_file.name.endswith(".csv") else pd.read_excel(merge_file)
                    st.write("Merge dataset preview:", merge_df.head())
                    
                    join_type = st.selectbox("Join type", ["inner", "left", "right", "outer"])
                    left_key = st.selectbox("Left key column", df.columns)
                    right_key = st.selectbox("Right key column", merge_df.columns)
                    
                    if st.button("Perform Merge", key= "3"):
                        merged = pd.merge(df, merge_df, left_on=left_key, right_on=right_key, how=join_type)
                        st.session_state[DF_KEY] = merged
                        st.success(f"Merged dataset shape: {merged.shape}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Merge failed: {e}")

    # ---------- data cleaning ----------
    with st.expander("Data Cleaning", expanded=False):
        tabs = st.tabs(["Missing Values", "Outliers", "Duplicates", "Data Types", "Text Cleaning", "Advanced Cleaning"])
        
        
        with tabs[0]:  # Missing Values
            st.subheader("Handle Missing Values")
            missing_cols = df.columns[df.isna().any()].tolist()
            
            if missing_cols:
                col = st.selectbox("Select column with missing values", missing_cols)
                missing_count = df[col].isna().sum()
                st.write(f"Missing values in {col}: {missing_count} ({missing_count/len(df)*100:.1f}%)")
                
                strategy = st.radio("Handling strategy", 
                                  ["Drop rows", "Fill with mean/median", "Fill with mode", 
                                   "Fill with custom value", "Interpolate", "KNN Imputation"])
                
                if strategy == "Drop rows":
                    if st.button("Drop rows with missing values", key= "4"):
                        st.session_state[DF_KEY] = df.dropna(subset=[col])
                        st.success(f"Dropped {missing_count} rows")
                        
                elif strategy == "Fill with mean/median":
                    if df[col].dtype in ['int64', 'float64']:
                        fill_val = st.selectbox("Fill with", ["mean", "median"])
                        if st.button("Fill missing values",key= "6"):
                            fill_value = df[col].mean() if fill_val == "mean" else df[col].median()
                            st.session_state[DF_KEY][col] = df[col].fillna(fill_value)
                            st.success(f"Filled with {fill_val}: {fill_value:.2f}")
                    else:
                        st.warning("Column must be numeric for mean/median imputation")
                        
                elif strategy == "Fill with mode":
                    if st.button("Fill with mode", "8"):
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
                        st.session_state[DF_KEY][col] = df[col].fillna(mode_val)
                        st.success(f"Filled with mode: {mode_val}")
                        
                elif strategy == "Fill with custom value":
                    custom_val = st.text_input("Custom value to fill")
                    if st.button("Fill with custom value", key= "9"):
                        st.session_state[DF_KEY][col] = df[col].fillna(custom_val)
                        st.success(f"Filled with: {custom_val}")
                        
                elif strategy == "Interpolate":
                    if df[col].dtype in ['int64', 'float64']:
                        if st.button("Interpolate missing values", key= "10"):
                            st.session_state[DF_KEY][col] = df[col].interpolate()
                            st.success("Applied interpolation")
                    else:
                        st.warning("Interpolation only works for numeric columns")
                        
                elif strategy == "KNN Imputation":
                    if st.button("Apply KNN Imputation (nearest neighbors)", key= "13"):
                        from sklearn.impute import KNNImputer
                        imputer = KNNImputer(n_neighbors=5)
                        numeric_cols = df.select_dtypes(include=np.number).columns
                        df_numeric = df[numeric_cols].copy()
                        imputed = imputer.fit_transform(df_numeric)
                        st.session_state[DF_KEY][numeric_cols] = imputed
                        st.success("Applied KNN imputation to all numeric columns")
            else:
                st.success("No missing values found!")
        
        with tabs[1]:  # Outliers
            st.subheader("Detect and Handle Outliers")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_cols:
                outlier_col = st.selectbox("Select numeric column for outlier detection", numeric_cols)
                
                # Outlier detection methods
                method = st.radio("Outlier detection method", 
                                ["IQR (Interquartile Range)", "Z-score", "Isolation Forest"])
                
                if method == "IQR (Interquartile Range)":
                    Q1 = df[outlier_col].quantile(0.25)
                    Q3 = df[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
                    
                elif method == "Z-score":
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(df[outlier_col].dropna()))
                    outliers = df[z_scores > 3]
                    
                elif method == "Isolation Forest":
                    from sklearn.ensemble import IsolationForest
                    clf = IsolationForest(contamination=0.1, random_state=42)
                    preds = clf.fit_predict(df[[outlier_col]].dropna())
                    outliers = df[preds == -1]
                
                st.write(f"Detected {len(outliers)} potential outliers")
                st.dataframe(outliers)
                
                # Outlier handling
                if len(outliers) > 0:
                    action = st.selectbox("Action on outliers", 
                                        ["Show only", "Remove outliers", "Cap outliers", "Winsorize"])
                    
                    if action == "Remove outliers" and st.button("Remove Outliers", key= "14"):
                        if method == "IQR (Interquartile Range)":
                            st.session_state[DF_KEY] = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                        st.success(f"Removed {len(outliers)} outliers")
                        
                    elif action == "Cap outliers" and st.button("Cap Outliers", key= "15"):
                        if method == "IQR (Interquartile Range)":
                            df_capped = df.copy()
                            df_capped[outlier_col] = np.where(df_capped[outlier_col] < lower_bound, lower_bound, 
                                                            np.where(df_capped[outlier_col] > upper_bound, upper_bound, 
                                                                    df_capped[outlier_col]))
                            st.session_state[DF_KEY] = df_capped
                            st.success("Outliers capped to IQR bounds")
            else:
                st.info("No numeric columns for outlier detection")
        
        with tabs[2]:  # Duplicates
            st.subheader("Handle Duplicate Rows")
            duplicates = df.duplicated().sum()
            st.write(f"Number of duplicate rows: {duplicates}")
            
            if duplicates > 0:
                if st.button("Remove Duplicate Rows", key= "17"):
                    st.session_state[DF_KEY] = df.drop_duplicates()
                    st.success(f"Removed {duplicates} duplicate rows")
            else:
                st.success("No duplicate rows found!")
        
        with tabs[3]:  # Data Types
            st.subheader("Change Data Types")
            col_to_convert = st.selectbox("Select column to convert", df.columns)
            current_type = str(df[col_to_convert].dtype)
            st.write(f"Current type: {current_type}")
            
            new_type = st.selectbox("Convert to", 
                                  ["Keep as is", "numeric", "category", "datetime", "string"])
            
            if st.button("Convert Data Type", key= "18") and new_type != "Keep as is":
                try:
                    if new_type == "numeric":
                        st.session_state[DF_KEY][col_to_convert] = pd.to_numeric(df[col_to_convert], errors='coerce')
                    elif new_type == "category":
                        st.session_state[DF_KEY][col_to_convert] = df[col_to_convert].astype('category')
                    elif new_type == "datetime":
                        st.session_state[DF_KEY][col_to_convert] = pd.to_datetime(df[col_to_convert], errors='coerce')
                    elif new_type == "string":
                        st.session_state[DF_KEY][col_to_convert] = df[col_to_convert].astype('string')
                    st.success(f"Converted {col_to_convert} to {new_type}")
                except Exception as e:
                    st.error(f"Conversion failed: {e}")
        
        with tabs[4]:  # Text Cleaning
            st.subheader("Text Data Cleaning")
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            
            if text_cols:
                text_col = st.selectbox("Select text column", text_cols)
                st.write("Sample values:", df[text_col].head().tolist())
                
                operations = st.multiselect("Text cleaning operations",
                                          ["Lowercase", "Remove punctuation", "Remove digits", 
                                           "Remove extra spaces", "Remove stopwords", "Stemming"])
                
                if st.button("Apply Text Cleaning", key= "20"):
                    cleaned = df[text_col].copy()
                    
                    if "Lowercase" in operations:
                        cleaned = cleaned.str.lower()
                    
                    if "Remove punctuation" in operations:
                        cleaned = cleaned.str.replace(r'[^\w\s]', '', regex=True)
                    
                    if "Remove digits" in operations:
                        cleaned = cleaned.str.replace(r'\d+', '', regex=True)
                    
                    if "Remove extra spaces" in operations:
                        cleaned = cleaned.str.strip().str.replace(r'\s+', ' ', regex=True)
                    
                    if "Remove stopwords" in operations:
                        try:
                            import nltk
                            nltk.download('stopwords', quiet=True)
                            from nltk.corpus import stopwords
                            stop_words = set(stopwords.words('english'))
                            cleaned = cleaned.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
                        except:
                            st.warning("NLTK not available for stopword removal")
                    
                    if "Stemming" in operations:
                        try:
                            from nltk.stem import PorterStemmer
                            stemmer = PorterStemmer()
                            cleaned = cleaned.apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
                        except:
                            st.warning("NLTK not available for stemming")
                    
                    st.session_state[DF_KEY][text_col] = cleaned
                    st.success("Applied text cleaning operations")
                    st.write("Cleaned sample:", cleaned.head().tolist())
            else:
                st.info("No text columns found for cleaning")

            with tabs[5]:  # Advanced Cleaning Tab
                st.subheader("Advanced Data Cleaning Operations")
                
                # GroupBy Operations
                st.markdown("#### GroupBy Operations")
                group_col = st.selectbox("Group by column", [None] + df.columns.tolist(), key = "Me")
                if group_col:
                    agg_col = st.selectbox("Aggregate column", [col for col in df.columns if col != group_col])
                    agg_func = st.selectbox("Aggregation function", ["mean", "sum", "count", "min", "max", "std"])
                    
                    if st.button("Apply GroupBy", key= "23"):
                        grouped = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
                        st.write("GroupBy Result:")
                        st.dataframe(grouped)
                
                # Custom Python Code Execution
                st.markdown("#### Custom Python Code")
                custom_code = st.text_area("Enter custom Python code for data transformation", 
                                        value="# Example: df['new_column'] = df['existing_column'] * 2\n# Use 'df' to reference your dataframe",
                                        height=150)
                
                if st.button("Execute Custom Code", key= "24"):
                    try:
                        # Create a safe environment for code execution
                        local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
                        global_vars = {}
                        exec(custom_code, global_vars, local_vars)
                        
                        if 'df' in local_vars:
                            st.session_state[DF_KEY] = local_vars['df']
                            st.success("Custom code executed successfully!")
                            st.rerun()
                        else:
                            st.warning("Code executed but dataframe 'df' was not modified.")
                    except Exception as e:
                        st.error(f"Error executing custom code: {e}")
                
                # Column Operations
                st.markdown("#### Column Operations")
                col_ops_col = st.selectbox("Select column for operation", df.columns.tolist())
                col_operation = st.selectbox("Operation", 
                                        ["Keep first N words", "Keep last N words", 
                                            "Remove first N words", "Remove last N words",
                                            "Extract numbers", "Extract text", "Custom regex"])
                
                if col_operation in ["Keep first N words", "Keep last N words", "Remove first N words", "Remove last N words"]:
                    n_words = st.number_input("Number of words", min_value=1, value=3, key = "1*")
                    if st.button("Apply Text Operation", key= "25"):
                        if col_operation == "Keep first N words":
                            st.session_state[DF_KEY][col_ops_col] = df[col_ops_col].astype(str).apply(
                                lambda x: ' '.join(x.split()[:n_words]))
                        elif col_operation == "Keep last N words":
                            st.session_state[DF_KEY][col_ops_col] = df[col_ops_col].astype(str).apply(
                                lambda x: ' '.join(x.split()[-n_words:]))
                        elif col_operation == "Remove first N words":
                            st.session_state[DF_KEY][col_ops_col] = df[col_ops_col].astype(str).apply(
                                lambda x: ' '.join(x.split()[n_words:]))
                        elif col_operation == "Remove last N words":
                            st.session_state[DF_KEY][col_ops_col] = df[col_ops_col].astype(str).apply(
                                lambda x: ' '.join(x.split()[:-n_words]))
                        st.success("Text operation applied!")
                
                elif col_operation == "Custom regex":
                    regex_pattern = st.text_input("Regex pattern", value=r"(\d+)")
                    if st.button("Apply Regex", key= "28"):
                        try:
                            st.session_state[DF_KEY][f"{col_ops_col}_extracted"] = df[col_ops_col].astype(str).str.extract(regex_pattern)
                            st.success("Regex extraction applied!")
                        except Exception as e:
                            st.error(f"Regex error: {e}")

    # ---------- feature engineering ----------
    with st.expander("Feature Engineering", expanded=False):
        st.subheader("Create New Features")
        
        # Create new column from existing
        col1, col2 = st.columns(2)
        with col1:
            new_col_name = st.text_input("New column name", "new_feature")
        with col2:
            operation = st.selectbox("Operation", 
                                   ["Custom expression", "Binning", "One-hot encoding", 
                                    "Date extraction", "Text length"])
        
        if operation == "Custom expression":
            expr = st.text_input("Pandas expression (use df to reference dataframe)", "df['col1'] + df['col2']")
            if st.button("Create Column",key= "29") and expr:
                try:
                    # Safe evaluation
                    allowed_globals = {'df': df, 'np': np, 'pd': pd}
                    result = eval(expr, {"__builtins__": {}}, allowed_globals)
                    st.session_state[DF_KEY][new_col_name] = result
                    st.success(f"Created column '{new_col_name}'")
                except Exception as e:
                    st.error(f"Error in expression: {e}")
        
        elif operation == "Binning":
            bin_col = st.selectbox("Column to bin", df.select_dtypes(include=np.number).columns.tolist())
            bin_method = st.radio("Binning method", ["Equal width", "Equal frequency", "Custom bins"])
            
            if bin_method == "Equal width":
                n_bins = st.slider("Number of bins", 2, 20, 5, key ="1*")
                if st.button("Create Bins", key= "30"):
                    st.session_state[DF_KEY][new_col_name] = pd.cut(df[bin_col], bins=n_bins, labels=False)
                    
            elif bin_method == "Equal frequency":
                n_bins = st.slider("Number of bins", 2, 20, 5,key ="2*")
                if st.button("Create Bins", key= "31"):
                    st.session_state[DF_KEY][new_col_name] = pd.qcut(df[bin_col], q=n_bins, labels=False, duplicates='drop')
                    
            elif bin_method == "Custom bins":
                bin_edges = st.text_input("Bin edges (comma separated)", "0,25,50,75,100")
                if st.button("Create Bins",key= "32"):
                    edges = [float(x.strip()) for x in bin_edges.split(',')]
                    st.session_state[DF_KEY][new_col_name] = pd.cut(df[bin_col], bins=edges, labels=False)
        
        elif operation == "One-hot encoding":
            cat_col = st.selectbox("Categorical column", df.select_dtypes(exclude=np.number).columns.tolist())
            if st.button("One-hot encode", key= "33"):
                dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
                st.session_state[DF_KEY] = pd.concat([df, dummies], axis=1)
                st.success(f"Created {len(dummies.columns)} one-hot encoded columns")
        
        elif operation == "Date extraction":
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if date_cols:
                date_col = st.selectbox("Date column", date_cols)
                extract = st.multiselect("Extract components", 
                                       ["Year", "Month", "Day", "Dayofweek", "Quarter", "Is weekend"])
                if st.button("Extract Date Components", key= "33"):
                    for comp in extract:
                        if comp == "Year":
                            st.session_state[DF_KEY][f"{date_col}_year"] = df[date_col].dt.year
                        elif comp == "Month":
                            st.session_state[DF_KEY][f"{date_col}_month"] = df[date_col].dt.month
                        elif comp == "Day":
                            st.session_state[DF_KEY][f"{date_col}_day"] = df[date_col].dt.day
                        elif comp == "Dayofweek":
                            st.session_state[DF_KEY][f"{date_col}_dayofweek"] = df[date_col].dt.dayofweek
                        elif comp == "Quarter":
                            st.session_state[DF_KEY][f"{date_col}_quarter"] = df[date_col].dt.quarter
                        elif comp == "Is weekend":
                            st.session_state[DF_KEY][f"{date_col}_is_weekend"] = df[date_col].dt.dayofweek.isin([5,6])
                    st.success(f"Extracted {len(extract)} date components")
            else:
                st.info("No datetime columns found")
        
        elif operation == "Text length":
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            if text_cols:
                text_col = st.selectbox("Text column", text_cols)
                if st.button("Create Text Length Feature", key= "34"):
                    st.session_state[DF_KEY][new_col_name] = df[text_col].str.len()
                    st.success("Created text length feature")

                    

    # ---------- visualization ----------
    with st.expander("Data Visualization", expanded=False):
        st.subheader("Interactive Visualizations")
        
        # Chart type selection
        chart_type = st.selectbox("Chart Type", 
                                ["Histogram", "Box", "Violin", "Scatter", "Line", "Area", 
                                 "Pie", "Density 2D", "Heatmap", "3D Scatter", "3D Surface",
                                 "Animated Scatter", "Animated Bar"])
        
        # Column selection based on chart type
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X Axis", [None] + list(df.columns))
        with col2:
            y_axis = st.selectbox("Y Axis", [None] + list(df.columns)) if chart_type not in ["Histogram", "Pie"] else None
        with col3:
            z_axis = st.selectbox("Z Axis", [None] + list(df.columns)) if chart_type in ["3D Scatter", "3D Surface"] else None
        
        # Additional options
        color_by = st.selectbox("Color by", [None] + list(df.columns))
        size_by = st.selectbox("Size by", [None] + list(df.columns)) if chart_type in ["Scatter", "3D Scatter"] else None
        
        # Advanced options
        with st.expander("Advanced Options"):
            facet_col = st.selectbox("Facet Column", [None] + list(df.columns))
            facet_row = st.selectbox("Facet Row", [None] + list(df.columns))
            trendline = st.selectbox("Trendline", [None, "ols", "lowess"]) if chart_type == "Scatter" else None
            animation_frame = st.selectbox("Animation Frame", [None] + list(df.columns)) if chart_type in ["Animated Scatter", "Animated Bar"] else None
        
        if st.button("Generate Chart", key= "90"):
            try:
                fig = make_chart(
                    df_=df,
                    kind=chart_type,
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
                    color=color_by,
                    size=size_by,
                    facet_col=facet_col,
                    facet_row=facet_row,
                    trendline=trendline,
                    animation_frame=animation_frame,
                    title=f"{chart_type} of {x_axis} vs {y_axis}" if y_axis else f"{chart_type} of {x_axis}"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart generation failed: {e}")

            # Advanced Visualization

    with st.expander("Advanced Visualization", expanded=False):
        tabs = st.tabs(["3D Charts", "Custom Chart Builder", "Python Code Charts"])
        
        with tabs[0]:  # 3D Charts
            st.subheader("3D Visualization")
            
            # 3D Scatter Plot
            st.markdown("#### 3D Scatter Plot")
            col1, col2, col3 = st.columns(3)
            with col1:
                x_3d = st.selectbox("X Axis", df.columns.tolist(), key="x_3d")
            with col2:
                y_3d = st.selectbox("Y Axis", df.columns.tolist(), key="y_3d")
            with col3:
                z_3d = st.selectbox("Z Axis", df.columns.tolist(), key="z_3d")
            
            color_3d = st.selectbox("Color by", [None] + df.columns.tolist(), key="color_3d")
            size_3d = st.selectbox("Size by", [None] + df.select_dtypes(include=np.number).columns.tolist(), key="size_3d")
            
            if st.button("Create 3D Scatter Plot", key= "36"):
                fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color=color_3d, size=size_3d,
                                  title=f"3D Scatter: {x_3d} vs {y_3d} vs {z_3d}")
                st.plotly_chart(fig, use_container_width=True)
            
            # 3D Surface Plot (for numeric matrices)
            st.markdown("#### 3D Surface Plot")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) >= 2:
                surf_x = st.selectbox("Surface X", num_cols, key="surf_x")
                surf_y = st.selectbox("Surface Y", num_cols, key="surf_y")
                surf_z = st.selectbox("Surface Z", num_cols, key="surf_z")
                
                if st.button("Create 3D Surface Plot", key= "37"):
                    try:
                        pivot_df = df.pivot_table(values=surf_z, index=surf_y, columns=surf_x, aggfunc='mean')
                        fig = go.Figure(data=[go.Surface(z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index)])
                        fig.update_layout(title=f"3D Surface: {surf_z} by {surf_x} and {surf_y}")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not create surface plot: {e}")
        
        with tabs[1]:  # Custom Chart Builder
            st.subheader("Custom Chart Builder")
            
            chart_library = st.selectbox("Chart Library", ["Plotly Express", "Matplotlib", "Seaborn"])
            chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Heatmap"])
            
            # Dynamic parameter selection based on chart type
            if chart_type in ["Scatter", "Line"]:
                x_var = st.selectbox("X Variable", df.columns.tolist())
                y_var = st.selectbox("Y Variable", df.columns.tolist())
                color_var = st.selectbox("Color Variable", [None] + df.columns.tolist(),key = "Le")
            
            if chart_type == "Histogram":
                hist_var = st.selectbox("Variable", df.columns.tolist())
                bins = st.slider("Number of bins", 5, 100, 30, key ="3")
            
            if st.button("Generate Custom Chart", key= "38"):
                try:
                    if chart_library == "Plotly Express":
                        if chart_type == "Scatter":
                            fig = px.scatter(df, x=x_var, y=y_var, color=color_var)
                        elif chart_type == "Line":
                            fig = px.line(df, x=x_var, y=y_var, color=color_var)
                        elif chart_type == "Bar":
                            fig = px.bar(df, x=x_var, y=y_var, color=color_var)
                        elif chart_type == "Histogram":
                            fig = px.histogram(df, x=hist_var, nbins=bins)
                        elif chart_type == "Box":
                            fig = px.box(df, x=x_var, y=y_var, color=color_var)
                        elif chart_type == "Violin":
                            fig = px.violin(df, x=x_var, y=y_var, color=color_var)
                        elif chart_type == "Heatmap":
                            num_df = df.select_dtypes(include=np.number)
                            corr = num_df.corr()
                            fig = px.imshow(corr, text_auto=True, aspect="auto")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_library == "Seaborn":
                        fig, ax = plt.subplots()
                        if chart_type == "Scatter":
                            sns.scatterplot(data=df, x=x_var, y=y_var, hue=color_var, ax=ax)
                        elif chart_type == "Line":
                            sns.lineplot(data=df, x=x_var, y=y_var, hue=color_var, ax=ax)
                        elif chart_type == "Bar":
                            sns.barplot(data=df, x=x_var, y=y_var, hue=color_var, ax=ax)
                        elif chart_type == "Histogram":
                            sns.histplot(data=df, x=hist_var, bins=bins, ax=ax)
                        elif chart_type == "Box":
                            sns.boxplot(data=df, x=x_var, y=y_var, hue=color_var, ax=ax)
                        elif chart_type == "Violin":
                            sns.violinplot(data=df, x=x_var, y=y_var, hue=color_var, ax=ax)
                        elif chart_type == "Heatmap":
                            num_df = df.select_dtypes(include=np.number)
                            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", ax=ax)
                        
                        st.pyplot(fig)
                    
                    elif chart_library == "Matplotlib":
                        fig, ax = plt.subplots()
                        if chart_type == "Scatter":
                            ax.scatter(df[x_var], df[y_var])
                            ax.set_xlabel(x_var)
                            ax.set_ylabel(y_var)
                        elif chart_type == "Line":
                            ax.plot(df[x_var], df[y_var])
                            ax.set_xlabel(x_var)
                            ax.set_ylabel(y_var)
                        elif chart_type == "Bar":
                            ax.bar(df[x_var], df[y_var])
                            ax.set_xlabel(x_var)
                            ax.set_ylabel(y_var)
                        elif chart_type == "Histogram":
                            ax.hist(df[hist_var], bins=bins)
                            ax.set_xlabel(hist_var)
                        elif chart_type == "Box":
                            ax.boxplot([df[df[color_var] == val][y_var] for val in df[color_var].unique()] if color_var else [df[y_var]])
                            ax.set_xticklabels(df[color_var].unique() if color_var else [y_var])
                        
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Chart generation failed: {e}")
        
        with tabs[2]:  # Python Code Charts
            st.subheader("Python Code Chart Generation")
            
            code_library = st.selectbox("Select Library", ["Plotly", "Seaborn", "Matplotlib", "Plotly+Graph_Objects"])
            
            code_template = ""
            if code_library == "Plotly":
                code_template = """import plotly.express as px

fig = px.scatter(df, x='column1', y='column2', color='column3')
fig.update_layout(title='Custom Chart')
fig.show()"""
            
            elif code_library == "Seaborn":
                code_template = """import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='column1', y='column2', hue='column3', ax=ax)
ax.set_title('Custom Chart')
plt.show()"""
            
            elif code_library == "Matplotlib":
                code_template = """import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['column1'], df['column2'], c=df['column3'], alpha=0.6)
ax.set_xlabel('Column 1')
ax.set_ylabel('Column 2')
ax.set_title('Custom Chart')
plt.colorbar(ax.collections[0], label='Column 3')
plt.show()"""
            
            elif code_library == "Plotly+Graph_Objects":
                code_template = """import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=df['column1'], y=df['column2'], mode='markers',
                         marker=dict(color=df['column3'], colorscale='Viridis')))
fig.update_layout(title='Custom Chart')
fig.show()"""
            
            custom_chart_code = st.text_area("Enter your chart code", value=code_template, height=200,
                                           help="Use 'df' to reference your dataframe. The code should create and display a chart.")
            
            if st.button("Execute Chart Code", key= "39"):
                try:
                    # Create a safe execution environment
                    local_vars = {'df': df, 'plt': plt, 'sns': sns, 'px': px, 'go': go, 'make_subplots': make_subplots}
                    exec(custom_chart_code, {}, local_vars)
                    
                    # Check if a figure was created
                    if 'fig' in local_vars:
                        fig = local_vars['fig']
                        if hasattr(fig, 'show'):
                            if hasattr(fig, '__plotly_restyle'):  # Plotly figure
                                st.plotly_chart(fig, use_container_width=True)
                            else:  # Matplotlib figure
                                st.pyplot(fig)
                        else:
                            st.warning("Code executed but no valid figure object found.")
                    else:
                        st.info("Code executed. If you created a chart, make sure it's assigned to variable 'fig'.")
                
                except Exception as e:
                    st.error(f"Error executing chart code: {e}")
                    st.error(f"Error details: {str(e)}")

    # ---------- correlation analysis ----------
    with st.expander("Correlation Analysis", expanded=False):
        st.subheader("Correlation Matrix")
        
        numeric_df = df.select_dtypes(include=np.number)
        if len(numeric_df.columns) > 1:
            corr_method = st.radio("Correlation method", ["Pearson", "Spearman", "Kendall"])
            
            if corr_method == "Pearson":
                corr_matrix = numeric_df.corr()
            elif corr_method == "Spearman":
                corr_matrix = numeric_df.corr(method='spearman')
            else:
                corr_matrix = numeric_df.corr(method='kendall')
            
            # Heatmap
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                          color_continuous_scale='RdBu_r', title=f"{corr_method} Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            # Find highly correlated pairs
            st.subheader("Highly Correlated Features")
            threshold = st.slider("Correlation threshold", 0.5, 1.0, 0.8, 0.05,key ="4")
            
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr:
                high_corr_df = pd.DataFrame(high_corr, columns=['Feature 1', 'Feature 2', 'Correlation'])
                st.dataframe(high_corr_df.sort_values('Correlation', ascending=False))
            else:
                st.info(f"No feature pairs with correlation > {threshold}")
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")

    # ---------- automated profiling ----------
    with st.expander("Automated Data Profiling", expanded=False):
        st.subheader("Generate Comprehensive Profile Report")
        
        if ProfileReport is None:
            st.warning("ydata_profiling is not installed. Install with: pip install ydata-profiling")
        else:
            if st.button("Generate Profile Report",key= "40"):
                with st.spinner("Generating comprehensive profile report..."):
                    profile = ProfileReport(df, title="Dataset Profile", explorative=True)
                    
                    # Save to HTML
                    profile_file = "data_profile.html"
                    profile.to_file(profile_file)
                    
                    # Display in Streamlit
                    with open(profile_file, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    
                    components.html(html_content, height=800, scrolling=True)
                    
                    # Download button
                    with open(profile_file, "rb") as f:
                        st.download_button("Download Profile Report", f, file_name="data_profile.html")

    # ---------- before/after comparison ----------
    if st.session_state.eda_snapshot:
        with st.expander("Before/After Comparison", expanded=False):
            label, snapshot_df = st.session_state.eda_snapshot
            st.subheader(f"Comparison: Current vs {label}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Rows", df.shape[0])
                st.metric("Current Columns", df.shape[1])
                st.metric("Current Missing", df.isna().sum().sum())
            with col2:
                st.metric("Snapshot Rows", snapshot_df.shape[0], 
                         delta=df.shape[0] - snapshot_df.shape[0])
                st.metric("Snapshot Columns", snapshot_df.shape[1],
                         delta=df.shape[1] - snapshot_df.shape[1])
                st.metric("Snapshot Missing", snapshot_df.isna().sum().sum(),
                         delta=df.isna().sum().sum() - snapshot_df.isna().sum().sum())
            
            # Show differences
            if not df.equals(snapshot_df):
                st.info("Datasets differ in content")
            else:
                st.success("No differences found")


    # ---------- bottom actions ----------
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Snapshot Current State", key= "41"):
            snapshot_current("manual_snapshot")
    
    with col2:
        if st.button("Reset to Original",key= "42"):
            if st.session_state.versions:
                original_df = st.session_state.versions[0][1]
                st.session_state[DF_KEY] = original_df.copy()
                st.success("Reset to original dataset")
    
    with col3:
        download_panel(df, filename_basename="cleaned_dataset")

    #st.warning("**Note:** All changes are applied to the current session. Download your cleaned dataset to preserve changes.")

################################################
# Page 3: Train-Test-Split
################################################

elif page == "Train-Test Split":
    st.header("Train-Test Split")
    ensure_session_state()
    inject_css()

    st.markdown("# Train-Test Split")
    st.caption("Split dataset into training, test (and optional validation) sets.")

    if st.session_state.df is None:
        st.warning("No dataset loaded. Please upload on Home / Data Loading page.")
        st.stop()

    df = st.session_state.df

    # ---------------- Select Target ----------------
    target_col = st.selectbox("Select target column", [None] + df.columns.tolist(), key = "Fe")
    if not target_col:
        st.info("Please select a target column to continue.")
        st.stop()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # AI Piece - Train Test Split
    if st.session_state.groq_available:
        with st.expander("🤖 AI Split Recommendations", expanded=False):
            if st.button("Get AI Split Recommendations", key="43"):
                analysis = groq_data_analysis(
                    "Based on this dataset, recommend optimal train-test split parameters including:\n"
                    "1. Recommended test size percentage and why\n"
                    "2. Whether to use validation set and what size\n"
                    "3. Any stratification considerations for imbalanced data\n"
                    "4. Random state recommendations",
                    st.session_state.df,
                    f"Target column: {target_col if target_col else 'Not selected yet'}"
                )
                st.info(analysis)

    # ---------------- Parameters ----------------
    st.markdown("### Split Parameters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        test_size = st.slider("Test size (%)", 5, 50, 20, step=5, key="6") / 100.0

    with col2:
        shuffle = st.checkbox("Shuffle", value=True, key="1")

    with col3:
        random_state = st.number_input("Random state", min_value=0, value=42, step=1, key="5*")

    with col4:
        use_val = st.checkbox("Also create Validation set", value=False, key="2")

    val_size = None
    if use_val:
        val_size = st.slider("Validation size (%) of train", 5, 50, 20, step=5, key="7*") / 100.0

    #-----------------------------------------------

    def detect_task_type_from_target(y):
        """Detect if target is for classification or regression"""
        if y is None or len(y) == 0:
            return "unknown"
        
        # Check if target looks like classification
        unique_values = y.nunique()
        if unique_values <= 15 or y.dtype == 'object' or y.dtype.name == 'category':
            return "classification"
        else:
            return "regression"

    # Use it in your page
    task_type = detect_task_type_from_target(y)

    # ---------------- Stratification Options ----------------
    st.markdown("### Stratification Options")

    # Check if stratification is appropriate - use y instead of y_train
    can_stratify = (task_type == "classification" and 
                    y.nunique() > 1 and 
                    len(y) > 0 and
                    not y.isna().any())

    if can_stratify:
        # Display class distribution info - use y instead of y_train
        class_distribution = y.value_counts()
        st.write("**Class distribution in full dataset:**")
        st.write(class_distribution)
        
        # Check if classes are imbalanced
        imbalance_ratio = class_distribution.max() / class_distribution.min()
        is_imbalanced = imbalance_ratio > 2.0  # Consider imbalanced if ratio > 2:1
        
        if is_imbalanced:
            st.warning(f"Class imbalance detected (ratio: {imbalance_ratio:.1f}:1). Stratification is recommended!")
        else:
            st.info("✓ Classes are relatively balanced")
        
        # Stratification option
        stratify_option = st.radio(
            "Stratification",
            options=["Auto (Recommended)", "Yes", "No"],
            index=0,
            help="Preserve class distribution in splits. Recommended for imbalanced data."
        )
        
        # Determine whether to stratify
        if stratify_option == "Auto (Recommended)":
            use_stratify = is_imbalanced or y.nunique() <= 10
        elif stratify_option == "Yes":
            use_stratify = True
        else:
            use_stratify = False
            
    else:
        use_stratify = False
        if task_type != "classification":
            st.info("Stratification is only available for classification problems")
        elif y.nunique() <= 1:  # Use y instead of y_train
            st.warning("Cannot stratify: Only one class present in target")
        elif y.isna().any():  # Use y instead of y_train
            st.warning("Cannot stratify: Target contains missing values")

    # Show stratification decision
    if can_stratify:
        if use_stratify:
            st.success("I Will use stratified sampling")
        else:
            st.info("I Will NOT use stratified sampling")

    # ---------------- Split Action ----------------
    if st.button("Perform Split", key="perform_split_btn"):
        try:
            # First split: train+val vs test
            split_kwargs = {
                "test_size": test_size,
                "shuffle": shuffle,
                "random_state": random_state
            }
            
            # Add stratification if appropriate
            if use_stratify:
                split_kwargs["stratify"] = y
            
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y, **split_kwargs
            )

            if use_val:
                # For validation split, we need to stratify again if needed
                val_split_kwargs = {
                    "test_size": val_size,
                    "shuffle": shuffle,
                    "random_state": random_state
                }
                
                if use_stratify:
                    val_split_kwargs["stratify"] = y_trainval
                
                # Split train into train/val
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval, y_trainval, **val_split_kwargs
                )
            else:
                X_train, y_train = X_trainval, y_trainval
                X_val, y_val = None, None

            # Store in session with timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state["split_result"] = {
                "timestamp": ts,
                "X_train": X_train, "y_train": y_train,
                "X_val": X_val, "y_val": y_val,
                "X_test": X_test, "y_test": y_test,
                "used_stratification": use_stratify,
                "class_distribution": {
                    "original": y.value_counts().to_dict(),
                    "train": y_train.value_counts().to_dict(),
                    "test": y_test.value_counts().to_dict()
                }
            }
            st.success(f"Split performed (saved as version {ts})")
            
            # Show class distribution comparison
            if use_stratify:
                st.markdown("### Class Distribution Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Original**")
                    st.write(y.value_counts())
                
                with col2:
                    st.write("**Training Set**")
                    st.write(y_train.value_counts())
                
                with col3:
                    st.write("**Test Set**")
                    st.write(y_test.value_counts())
                
                # Check if stratification worked well
                original_ratios = y.value_counts(normalize=True)
                train_ratios = y_train.value_counts(normalize=True)
                test_ratios = y_test.value_counts(normalize=True)
                
                # Calculate similarity
                similarity = min(
                    sum(abs(original_ratios - train_ratios)),
                    sum(abs(original_ratios - test_ratios))
                )
                
                if similarity < 0.1:  # Good stratification
                    st.success("✓ Stratification successful: Class distributions preserved")
                else:
                    st.warning("Stratification may not have worked perfectly")

        except ValueError as e:
            if "least populated class" in str(e):
                st.error("Stratification failed: Some classes have too few samples")
                st.error("Try reducing test size or disabling stratification")
            else:
                st.error(f"Split failed: {e}")

    # ---------------- AI Stratification Advice ----------------
    if st.session_state.groq_available and can_stratify:
        with st.expander("🤖 AI Stratification Advice"):
            if st.button("Get AI Stratification Recommendation", key="ai_stratify_help"):
                # Create the full prompt properly
                full_prompt = f"""
                Context information:
                Task type: {task_type}
                Number of classes: {y.nunique()}
                Class distribution: {class_distribution.to_dict()}
                Imbalance ratio: {imbalance_ratio:.1f}
                Dataset size: {len(y)} samples
                Test size: {test_size*100}%
                
                Question: Should I use stratification for this train-test split? 
                Consider the class distribution, imbalance ratio, and dataset size. 
                Provide specific advice and explain the reasoning.
                """
                
                advice = call_groq_llm(full_prompt)
                st.info("🤖 Stratification Advice:")
                st.write(advice)


    # ---------------- Preview ----------------
    if "split_result" in st.session_state:
        split = st.session_state["split_result"]

        st.markdown("### Split Summary")
        summary = {
            "Set": ["Train", "Validation" if split["X_val"] is not None else "—", "Test"],
            "Rows": [
                len(split["X_train"]),
                len(split["X_val"]) if split["X_val"] is not None else "—",
                len(split["X_test"]),
            ],
            "Features": [split["X_train"].shape[1]] * 3,
        }
        
        # Add stratification info if available
        if "used_stratification" in split:
            summary["Stratified"] = [
                "Yes" if split["used_stratification"] else "No",
                "Yes" if split["used_stratification"] else "No", 
                "Yes" if split["used_stratification"] else "No"
            ]
        
        st.dataframe(pd.DataFrame(summary))
        
        # Show detailed class distribution if stratified
        if split.get("used_stratification", False) and "class_distribution" in split:
            with st.expander("View Detailed Class Distribution"):
                st.write("**Original data:**")
                st.write(split["class_distribution"]["original"])
                
                st.write("**Training set:**")
                st.write(split["class_distribution"]["train"])
                
                st.write("**Test set:**")
                st.write(split["class_distribution"]["test"])
                
                # Calculate and show preservation ratios
                orig = pd.Series(split["class_distribution"]["original"])
                train = pd.Series(split["class_distribution"]["train"])
                test = pd.Series(split["class_distribution"]["test"])
                
                train_preservation = (train / train.sum()) / (orig / orig.sum())
                test_preservation = (test / test.sum()) / (orig / orig.sum())
                
                st.write("**Class preservation ratios (1.0 = perfect preservation):**")
                st.write("Training set:", train_preservation.to_dict())
                st.write("Test set:", test_preservation.to_dict())

        # ---------------- Downloads ----------------
        st.markdown("### Download Splits")

        def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
            return df.to_csv(index=False).encode("utf-8")

        cols = st.columns(3)
        with cols[0]:
            st.download_button("X_train.csv", df_to_csv_bytes(split["X_train"]), file_name="X_train.csv", mime="text/csv")
            st.download_button("y_train.csv", df_to_csv_bytes(split["y_train"].to_frame()), file_name="y_train.csv", mime="text/csv")
        with cols[1]:
            if split["X_val"] is not None:
                st.download_button("X_val.csv", df_to_csv_bytes(split["X_val"]), file_name="X_val.csv", mime="text/csv")
                st.download_button("y_val.csv", df_to_csv_bytes(split["y_val"].to_frame()), file_name="y_val.csv", mime="text/csv")
        with cols[2]:
            st.download_button("X_test.csv", df_to_csv_bytes(split["X_test"]), file_name="X_test.csv", mime="text/csv")
            st.download_button("y_test.csv", df_to_csv_bytes(split["y_test"].to_frame()), file_name="y_test.csv", mime="text/csv")

        # ZIP download
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for k in ["X_train","y_train","X_val","y_val","X_test","y_test"]:
                if split[k] is not None:
                    data = split[k].to_csv(index=False).encode("utf-8") if isinstance(split[k], pd.DataFrame) else split[k].to_frame().to_csv(index=False).encode("utf-8")
                    z.writestr(f"{k}.csv", data)
        st.download_button("Download All as ZIP", data=buf.getvalue(), file_name=f"splits_{split['timestamp']}.zip", mime="application/zip")

    st.warning("Remember to download your splits to keep them safe.")


# ################################################
# # Page: Pipeline and Model Training (Merged)
# ################################################

# elif page == "Pipeline and Model Training":
#     st.header("Pipeline and Model Training")
#     ensure_session_state()
#     inject_css()
#     init_groq_client()

#     st.markdown("# Pipeline and Model Training")
#     st.caption("Build preprocessing pipelines, select models, and configure hyperparameter tuning strategies")

#     if st.session_state.df is None:
#         st.warning("No dataset loaded. Please upload and split data first.")
#         st.stop()

#     # Get splits
#     splits = st.session_state.get("split_result")
#     if not splits:
#         st.warning("Please perform Train-Test Split first.")
#         st.stop()

#     X_train, X_test = splits["X_train"], splits["X_test"]
#     y_train, y_test = splits["y_train"], splits["y_test"]

#     cols = X_train.columns.tolist()

#     # --- Data Validation Section ---
#     st.markdown("## Data Validation")

#     original_df = st.session_state.df
#     all_columns = original_df.columns.tolist()

#     if 'target_column' not in st.session_state:
#         target_col = st.selectbox("Select target column", [None] + all_columns)
#         if target_col:
#             st.session_state.target_column = target_col
#         else:
#             st.info("Please select a target column to continue.")
#             st.stop()
#     else:
#         target_col = st.session_state.target_column

#     # Check if target variable is accidentally in features
#     if target_col in X_train.columns:
#         st.error(f"❌ CRITICAL ERROR: Target variable '{target_col}' is in the feature columns!")
#         st.error("This will cause perfect accuracy (1.0) because the model can see the answers.")
#         st.stop()

#     # Check for single class in target
#     if y_train.nunique() == 1:
#         st.error(f"Only one class found in target variable: {y_train.unique()[0]}")
#         st.error("This will always result in perfect accuracy for that class.")
#         st.stop()

#     # --- Column assignment ---
#     st.markdown("## Assign Columns")
#     num_cols = st.multiselect("Numeric Columns", cols, default=X_train.select_dtypes(include=np.number).columns.tolist())
#     cat_cols = st.multiselect("Categorical Columns", [c for c in cols if c not in num_cols], default=X_train.select_dtypes(exclude=np.number).columns.tolist())

#     st.info(f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

#     # --- Numeric Pipeline Builder ---
#     st.markdown("## Numeric Pipeline")
#     num_imputer = st.selectbox("Imputer", ["Mean", "Median", "Most Frequent", "Constant", "KNN", "Drop Rows", "None"])
#     num_scaler = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])

#     num_steps = []
#     if num_imputer == "Mean":
#         num_steps.append(("imputer", SimpleImputer(strategy="mean")))
#     elif num_imputer == "Median":
#         num_steps.append(("imputer", SimpleImputer(strategy="median")))
#     elif num_imputer == "Most Frequent":
#         num_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
#     elif num_imputer == "Constant":
#         num_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0)))
#     elif num_imputer == "KNN":
#         num_steps.append(("imputer", KNNImputer(n_neighbors=5)))
#     elif num_imputer == "Drop Rows":
#         X_train = X_train.dropna()
#         y_train = y_train.loc[X_train.index]
#         X_test = X_test.dropna()
#         y_test = y_test.loc[X_test.index]

#     if num_scaler == "StandardScaler":
#         num_steps.append(("scaler", StandardScaler()))
#     elif num_scaler == "MinMaxScaler":
#         num_steps.append(("scaler", MinMaxScaler()))
#     elif num_scaler == "RobustScaler":
#         num_steps.append(("scaler", RobustScaler()))

#     num_pipeline = Pipeline(num_steps) if num_steps else "passthrough"

#     # --- Categorical Pipeline Builder ---
#     st.markdown("## Categorical Pipeline")
#     cat_imputer = st.selectbox("Imputer", ["Most Frequent", "Constant", "None"])
#     cat_encoder = st.selectbox("Encoder", ["Ordinal", "OneHot", "None"])

#     cat_steps = []
#     if cat_imputer == "Most Frequent":
#         cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
#     elif cat_imputer == "Constant":
#         cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="missing")))

#     if cat_encoder == "Ordinal":
#         cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
#     elif cat_encoder == "OneHot":
#         cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
#     elif cat_encoder == "None":
#         st.warning("Categorical features must be numeric for most models. Falling back to OrdinalEncoder.")
#         cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

#     cat_pipeline = Pipeline(cat_steps) if cat_steps else "passthrough"

#     # --- Build final preprocessor ---
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", num_pipeline, num_cols),
#             ("cat", cat_pipeline, cat_cols)
#         ],
#         remainder="passthrough"
#     )

#     st.success("Preprocessor ready. Now choose models below.")

#     # Replace the current model selection code with this:

#     # --- Model Selection ---
#     st.markdown("## Model Selection")

#     def detect_task_type(y):
#         """Better task type detection"""
#         if pd.api.types.is_numeric_dtype(y):
#             unique_values = y.nunique()
#             if unique_values <= 15:
#                 unique_vals = sorted(y.dropna().unique())
#                 if (all(isinstance(v, (int, np.integer)) for v in unique_vals) and
#                     all(v in range(len(unique_vals)) for v in unique_vals)):
#                     return "classification"
#                 else:
#                     return "regression"
#             else:
#                 return "regression"
#         else:
#             return "classification"

#     task_type = detect_task_type(y_train)
#     st.info(f"Detected Task Type: **{task_type}**")

#     # Model options
#     models = {}

#     if task_type == "classification":
#         models = {
#             "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
#             "Random Forest": RandomForestClassifier(random_state=42),
#             "Gradient Boosting": GradientBoostingClassifier(random_state=42),
#             "SVM": SVC(probability=True, random_state=42),
#             "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=42),
#             "LightGBM": lgb.LGBMClassifier(random_state=42),
#             "K-Nearest Neighbors": KNeighborsClassifier(),
#             "Decision Tree": DecisionTreeClassifier(random_state=42),
#             "AdaBoost": AdaBoostClassifier(random_state=42),
#             "Naive Bayes": GaussianNB(),
#             "Neural Network (MLP) - Small": MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=1000),
#             "Neural Network (MLP) - Medium": MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
#             "Neural Network (MLP) - Large": MLPClassifier(hidden_layer_sizes=(100, 100, 50), random_state=42, max_iter=1000)
#         }
#     else:
#         models = {
#             "Linear Regression": LinearRegression(),
#             "Ridge": Ridge(alpha=1.0, random_state=42),
#             "Lasso": Lasso(alpha=1.0, random_state=42),
#             "Random Forest": RandomForestRegressor(random_state=42),
#             "Gradient Boosting": GradientBoostingRegressor(random_state=42),
#             "SVR": SVR(C=1.0, kernel='rbf'),
#             "XGBoost": xgb.XGBRegressor(random_state=42),
#             "LightGBM": lgb.LGBMRegressor(random_state=42),
#             "K-Nearest Neighbors": KNeighborsRegressor(),
#             "Decision Tree": DecisionTreeRegressor(random_state=42),
#             "AdaBoost": AdaBoostRegressor(random_state=42),
#             "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
#             "Neural Network (MLP) - Small": MLPRegressor(hidden_layer_sizes=(50,), random_state=42, max_iter=1000),
#             "Neural Network (MLP) - Medium": MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
#             "Neural Network (MLP) - Large": MLPRegressor(hidden_layer_sizes=(100, 100, 50), random_state=42, max_iter=1000)
#         }

#     # Change from selectbox to multiselect
#     selected_model_names = st.multiselect(
#         "Select Model(s)", 
#         list(models.keys()),
#         default=[list(models.keys())[0]] if models else []
#     )

#     selected_models = {name: models[name] for name in selected_model_names}

#         # --- Hyperparameter Tuning Strategy ---
#     st.markdown("## Hyperparameter Tuning Strategy")
    
#     # Initialize session state for tuning strategy
#     if 'tuning_strategy' not in st.session_state:
#         st.session_state.tuning_strategy = "default"
#     if 'custom_params' not in st.session_state:
#         st.session_state.custom_params = {}
#     if 'tuning_results' not in st.session_state:
#         st.session_state.tuning_results = {}
    
#     tuning_strategy = st.radio(
#         "Tuning Strategy",
#         ["Use Default Parameters", "Manual Customization", "Use Automated Tuning"],
#         key="tuning_strategy_radio"
#     )
    
#     # Map radio selection to strategy values
#     if tuning_strategy == "Use Default Parameters":
#         strategy_value = "default"
#     elif tuning_strategy == "Manual Customization":
#         strategy_value = "manual"
#     else:
#         strategy_value = "tune"
    
#     st.session_state.tuning_strategy = strategy_value
    
#     # Default parameter grids
#     default_param_grids = {
#         "Logistic Regression": {
#             'model__C': [0.1, 1, 10],
#             'model__penalty': ['l2', 'none'],
#             'model__solver': ['lbfgs', 'saga']
#         },
#         "Random Forest": {
#             'model__n_estimators': [50, 100, 200],
#             'model__max_depth': [3, 5, 10, None],
#             'model__min_samples_split': [2, 5, 10]
#         },
#         "Gradient Boosting": {
#             'model__n_estimators': [50, 100, 200],
#             'model__learning_rate': [0.01, 0.1, 0.2],
#             'model__max_depth': [3, 5, 7]
#         },
#         "SVM": {
#             'model__C': [0.1, 1, 10],
#             'model__kernel': ['linear', 'rbf'],
#             'model__gamma': ['scale', 'auto']
#         },
#         "XGBoost": {
#             'model__n_estimators': [50, 100, 200],
#             'model__max_depth': [3, 5, 7],
#             'model__learning_rate': [0.01, 0.1, 0.2]
#         }
#     }
    
#     # Manual Customization Section - Only show if models are selected
#     if strategy_value == "manual" and selected_model_names:
#         st.markdown("### Manual Hyperparameter Customization")
        
#         # Let user select which model to customize
#         model_to_customize = st.selectbox(
#             "Select model to customize parameters",
#             options=selected_model_names,
#             key="model_customize_select"
#         )
        
#         # Initialize custom params if not exists for this model
#         if model_to_customize not in st.session_state.custom_params:
#             st.session_state.custom_params[model_to_customize] = default_param_grids.get(model_to_customize, {}).copy()
        
#         custom_params = st.session_state.custom_params[model_to_customize]
        
#         # Display editable parameters if this model has default parameters
#         if model_to_customize in default_param_grids:
#             default_params = default_param_grids[model_to_customize]
            
#             for param_name, param_values in default_params.items():
#                 clean_name = param_name.replace('model__', '').replace('_', ' ').title()
                
#                 if isinstance(param_values[0], (int, float)):
#                     # For numeric parameters, use slider
#                     min_val = min(param_values)
#                     max_val = max(param_values)
#                     default_val = param_values[len(param_values)//2]
                    
#                     # Get current value or use default
#                     current_val = custom_params.get(param_name, [default_val])[0]
                    
#                     custom_val = st.slider(
#                         clean_name,
#                         min_value=min_val,
#                         max_value=max_val,
#                         value=current_val,
#                         key=f"manual_{model_to_customize}_{param_name}"
#                     )
#                     custom_params[param_name] = [custom_val]
#                 else:
#                     # For categorical parameters, use selectbox
#                     default_idx = 0
#                     if param_name in custom_params:
#                         current_val = custom_params[param_name][0]
#                         if current_val in param_values:
#                             default_idx = param_values.index(current_val)
                    
#                     custom_val = st.selectbox(
#                         clean_name,
#                         options=param_values,
#                         index=default_idx,
#                         key=f"manual_{model_to_customize}_{param_name}"
#                     )
#                     custom_params[param_name] = [custom_val]
            
#             if st.button("Save Parameters", key="save_params"):
#                 st.session_state.custom_params[model_to_customize] = custom_params
#                 st.success(f"Parameters saved for {model_to_customize}!")
#         else:
#             st.info(f"No default parameters available for {model_to_customize}. Using default model parameters.")
    
#     # Automated Tuning Section - Only show if models are selected
#     elif strategy_value == "tune" and selected_model_names:
#         st.markdown("### Automated Tuning Techniques")
        
#         # Let user select which model to tune
#         model_to_tune = st.selectbox(
#             "Select model to tune",
#             options=selected_model_names,
#             key="model_tune_select"
#         )
        
#         tuning_method = st.selectbox(
#             "Tuning Method",
#             ["GridSearchCV", "RandomizedSearchCV", "Bayesian Optimization (Optuna)"],
#             key="tuning_method"
#         )
        
#         cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5, key="cv_folds")
        
#         if st.button("Run Tuning", key="run_tuning"):
#             with st.spinner(f"Running {tuning_method} for {model_to_tune}..."):
#                 try:
#                     # Get the selected model
#                     selected_model = models[model_to_tune]
#                     default_params = default_param_grids.get(model_to_tune, {})
                    
#                     # Create pipeline
#                     pipe = Pipeline([
#                         ("preprocessor", preprocessor),
#                         ("model", selected_model)
#                     ])
                    
#                     if tuning_method == "GridSearchCV":
#                         search = GridSearchCV(
#                             pipe, 
#                             default_params, 
#                             cv=cv_folds,
#                             scoring='accuracy' if task_type == 'classification' else 'r2',
#                             n_jobs=-1,
#                             return_train_score=True
#                         )
#                         search.fit(X_train, y_train)
                        
#                     elif tuning_method == "RandomizedSearchCV":
#                         search = RandomizedSearchCV(
#                             pipe,
#                             default_params,
#                             n_iter=10,
#                             cv=cv_folds,
#                             scoring='accuracy' if task_type == 'classification' else 'r2',
#                             n_jobs=-1,
#                             random_state=42,
#                             return_train_score=True
#                         )
#                         search.fit(X_train, y_train)
                        
#                     else:  # Bayesian Optimization with Optuna
#                         def objective(trial):
#                             # Define parameter space for Optuna
#                             params = {}
#                             for param_name, param_values in default_params.items():
#                                 clean_name = param_name.replace('model__', '')
                                
#                                 if isinstance(param_values[0], (int, np.integer)):
#                                     params[param_name] = trial.suggest_int(clean_name, min(param_values), max(param_values))
#                                 elif isinstance(param_values[0], float):
#                                     params[param_name] = trial.suggest_float(clean_name, min(param_values), max(param_values))
#                                 else:
#                                     params[param_name] = trial.suggest_categorical(clean_name, param_values)
                            
#                             # Create model with suggested parameters
#                             model_clone = clone(selected_model)
#                             model_clone.set_params(**{k.replace('model__', ''): v for k, v in params.items()})
                            
#                             pipe = Pipeline([
#                                 ("preprocessor", preprocessor),
#                                 ("model", model_clone)
#                             ])
                            
#                             return cross_val_score(pipe, X_train, y_train, cv=cv_folds, 
#                                                  scoring='accuracy' if task_type == 'classification' else 'r2').mean()
                        
#                         study = optuna.create_study(direction="maximize")
#                         study.optimize(objective, n_trials=20)
                        
#                         # Convert Optuna results to match sklearn format
#                         search = type('obj', (object,), {
#                             'best_params_': study.best_params,
#                             'best_score_': study.best_value,
#                             'cv_results_': None
#                         })()
                    
#                     # Store results
#                     st.session_state.tuning_results[model_to_tune] = {
#                         'best_params': search.best_params_,
#                         'best_score': search.best_score_,
#                         'method': tuning_method
#                     }
                    
#                     st.success("Tuning completed successfully!")
                    
#                 except Exception as e:
#                     st.error(f"Tuning failed: {str(e)}")
        
#         # Display tuning results if available for this model
#         if model_to_tune in st.session_state.tuning_results:
#             results = st.session_state.tuning_results[model_to_tune]
            
#             st.markdown("#### Tuning Results")
#             st.json(results['best_params'])
#             st.write(f"Best Score: {results['best_score']:.4f}")
#             st.write(f"Method: {results['method']}")
            
#             if st.button("Apply Best Parameters", key="apply_best_params"):
#                 # Convert best parameters to manual customization format
#                 best_params = results['best_params']
#                 formatted_params = {}
                
#                 for param_name, param_value in best_params.items():
#                     formatted_params[param_name] = [param_value]
                
#                 st.session_state.custom_params[model_to_tune] = formatted_params
#                 st.session_state.tuning_strategy = "manual"
#                 st.success("Best parameters applied! Switch to Manual Customization to see them.")
    
#     # --- AI Assistant Integration ---
#     if st.session_state.groq_available:
#         with st.expander("🤖 AI Assistant", expanded=False):
#             st.markdown("Get expert advice on hyperparameter tuning")
            
#             # Context information for the AI
#             context = f"""
#             Dataset: {st.session_state.get('dataset_name', 'Unknown')}
#             Shape: {X_train.shape}
#             Task Type: {task_type}
#             Tuning Strategy: {tuning_strategy}
#             Numeric Features: {num_cols}
#             Categorical Features: {cat_cols}
#             """
            
#             question = st.text_input(
#                 "Ask the AI assistant about hyperparameter tuning:",
#                 placeholder="e.g., 'What are good starting values for Random Forest?', 'How should I tune learning rate?'",
#                 key="ai_question"
#             )
            
#             if st.button("Get AI Advice", key="ai_advice"):
#                 if question:
#                     with st.spinner("Consulting AI expert..."):
#                         prompt = f"""
#                         You are a machine learning expert. Provide specific, actionable advice about hyperparameter tuning.
                        
#                         Context:
#                         {context}
                        
#                         Question: {question}
                        
#                         Please provide:
#                         1. Specific parameter recommendations based on the dataset characteristics
#                         2. Explanation of why these values might work well
#                         3. Any warnings or considerations
#                         4. Suggested tuning strategy if applicable
                        
#                         Keep the response concise and practical.
#                         """
                        
#                         response = call_groq_llm(prompt)
#                         st.info(response)
#                 else:
#                     st.warning("Please enter a question first.")
    
#     # --- Training Execution ---
#     st.markdown("## Training Control")

#     # Determine which parameters to use for each model
#     models_params = {}
#     for model_name in selected_model_names:
#         if strategy_value == "default":
#             models_params[model_name] = default_param_grids.get(model_name, {})
#         elif strategy_value == "manual" and model_name in st.session_state.custom_params:
#             models_params[model_name] = st.session_state.custom_params[model_name]
#         else:
#             models_params[model_name] = {}

#     if st.button("Train Selected Models", key="train_model_final", type="primary"):
#         if not selected_model_names:
#             st.warning("Please select at least one model to train.")
#         else:
#             for model_name in selected_model_names:
#                 try:
#                     model = models[model_name]
#                     params_to_use = models_params[model_name]
                    
#                     # Create pipeline
#                     pipe = Pipeline([
#                         ("preprocessor", preprocessor),
#                         ("model", model)
#                     ])
                    
#                     # Set parameters if any are specified
#                     if params_to_use:
#                         pipe.set_params(**params_to_use)
                    
#                     # Train the model
#                     with st.spinner(f"Training {model_name}..."):
#                         start_time = time.time()
#                         pipe.fit(X_train, y_train)
#                         training_time = time.time() - start_time
                    
#                     # Make predictions
#                     y_pred = pipe.predict(X_test)
                    
#                     # Calculate metrics
#                     if task_type == "classification":
#                         acc = accuracy_score(y_test, y_pred)
#                         f1 = f1_score(y_test, y_pred, average="weighted")
#                         prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
#                         rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                        
#                         metrics = {
#                             "Accuracy": acc, 
#                             "Precision": prec, 
#                             "Recall": rec, 
#                             "F1": f1,
#                             "Training Time (s)": training_time
#                         }
#                     else:
#                         r2 = r2_score(y_test, y_pred)
#                         mae = mean_absolute_error(y_test, y_pred)
#                         mse = mean_squared_error(y_test, y_pred)
#                         rmse = np.sqrt(mse)
                        
#                         metrics = {
#                             "R²": r2, 
#                             "MAE": mae, 
#                             "MSE": mse, 
#                             "RMSE": rmse,
#                             "Training Time (s)": training_time
#                         }
                    
#                     # Store results
#                     if 'trained_models' not in st.session_state:
#                         st.session_state.trained_models = {}
#                     if 'model_results' not in st.session_state:
#                         st.session_state.model_results = {}
                    
#                     st.session_state.trained_models[model_name] = pipe
#                     st.session_state.model_results[model_name] = metrics
                    
#                     st.success(f"{model_name} trained successfully!")
                    
#                 except Exception as e:
#                     st.error(f"Training failed for {model_name}: {str(e)}")
            
#             # Determine best model after training all
#             if 'model_results' in st.session_state and st.session_state.model_results:
#                 best_model_name = None
#                 best_metric_value = -float('inf') if task_type == "classification" else float('inf')
                
#                 for model_name, metrics in st.session_state.model_results.items():
#                     primary_metric = next(iter(metrics.values()))
#                     if (task_type == "classification" and primary_metric > best_metric_value) or \
#                     (task_type == "regression" and primary_metric < best_metric_value):
#                         best_metric_value = primary_metric
#                         best_model_name = model_name
                
#                 if best_model_name:
#                     st.session_state.best_model = {
#                         "name": best_model_name,
#                         "pipeline": st.session_state.trained_models[best_model_name],
#                         "metrics": st.session_state.model_results[best_model_name]
#                     }
#                     st.success(f"🎯 Best model: {best_model_name}")
    
#     # Display final parameters that will be used
#     if params_to_use:
#         st.markdown("### Final Parameters")
#         try:
#             # Handle different parameter structures
#             if isinstance(params_to_use, dict) and all(isinstance(v, list) for v in params_to_use.values()):
#                 # Standard case: {'param1': [value1], 'param2': [value2]}
#                 params_df = pd.DataFrame.from_dict(params_to_use, orient='index', columns=['Value'])
#             elif isinstance(params_to_use, dict):
#                 # Case where values are not lists: {'param1': value1, 'param2': value2}
#                 params_df = pd.DataFrame.from_dict(params_to_use, orient='index', columns=['Value'])
#             else:
#                 # Fallback for unexpected structures
#                 params_df = pd.DataFrame({'Parameter': ['Custom parameters'], 'Value': ['Configured']})
            
#             st.dataframe(params_df)
#         except Exception as e:
#             st.warning(f"Could not display parameters: {e}")
#             st.write("Parameters:", params_to_use)
    
#     if st.button("Train Model", key="train_model_final", type="primary"):
#         try:
#             # Create pipeline
#             pipe = Pipeline([
#                 ("preprocessor", preprocessor),
#                 ("model", selected_model)
#             ])
            
#             # Set parameters if any are specified
#             if params_to_use:
#                 pipe.set_params(**params_to_use)
            
#             # Train the model
#             with st.spinner("Training model..."):
#                 start_time = time.time()
#                 pipe.fit(X_train, y_train)
#                 training_time = time.time() - start_time
            
#             # Make predictions
#             y_pred = pipe.predict(X_test)
            
#             # Calculate metrics
#             if task_type == "classification":
#                 acc = accuracy_score(y_test, y_pred)
#                 f1 = f1_score(y_test, y_pred, average="weighted")
#                 prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
#                 rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                
#                 metrics = {
#                     "Accuracy": acc, 
#                     "Precision": prec, 
#                     "Recall": rec, 
#                     "F1": f1,
#                     "Training Time (s)": training_time
#                 }
#             else:
#                 r2 = r2_score(y_test, y_pred)
#                 mae = mean_absolute_error(y_test, y_pred)
#                 mse = mean_squared_error(y_test, y_pred)
#                 rmse = np.sqrt(mse)
                
#                 metrics = {
#                     "R²": r2, 
#                     "MAE": mae, 
#                     "MSE": mse, 
#                     "RMSE": rmse,
#                     "Training Time (s)": training_time
#                 }
            
#             # Store results
#             if 'trained_models' not in st.session_state:
#                 st.session_state.trained_models = {}
#             if 'model_results' not in st.session_state:
#                 st.session_state.model_results = {}
            
#             st.session_state.trained_models[selected_model_name] = pipe
#             st.session_state.model_results[selected_model_name] = metrics
            
#             # Display results
#             st.success("Model trained successfully!")
#             st.markdown("### Performance Metrics")
#             metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
#             st.dataframe(metrics_df.style.format("{:.4f}"))
            
#             # Determine if this is the best model
#             if 'best_model' not in st.session_state:
#                 st.session_state.best_model = {
#                     "name": selected_model_name,
#                     "pipeline": pipe,
#                     "metrics": metrics
#                 }
#             else:
#                 # Compare with current best model
#                 current_best_metric = next(iter(st.session_state.best_model["metrics"].values()))
#                 new_metric = next(iter(metrics.values()))
                
#                 if (task_type == "classification" and new_metric > current_best_metric) or \
#                    (task_type == "regression" and new_metric < current_best_metric):
#                     st.session_state.best_model = {
#                         "name": selected_model_name,
#                         "pipeline": pipe,
#                         "metrics": metrics
#                     }
#                     st.success("🎯 New best model!")
            
#         except Exception as e:
#             st.error(f"Training failed: {str(e)}")
#             st.error(f"Error details: {str(e)}")
    
#     # --- Model Comparison ---
#     if 'model_results' in st.session_state and st.session_state.model_results:
#         st.markdown("## Model Comparison")
        
#         comparison_data = []
#         for model_name, metrics in st.session_state.model_results.items():
#             row = {"Model": model_name}
#             row.update(metrics)
#             if 'best_model' in st.session_state and st.session_state.best_model["name"] == model_name:
#                 row["Best"] = "⭐"
#             else:
#                 row["Best"] = ""
#             comparison_data.append(row)
        
#         comparison_df = pd.DataFrame(comparison_data)
        
#         # Style the comparison table
#         numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns.tolist()
#         if numeric_cols:
#             styled_df = comparison_df.style.highlight_max(
#                 subset=[col for col in numeric_cols if col != "Training Time (s)"], 
#                 color='lightgreen'
#             ).highlight_min(
#                 subset=["Training Time (s)"], 
#                 color='lightgreen'
#             )
            
#             st.dataframe(styled_df, use_container_width=True)

################################################
# Page: Pipeline and Model Training (Merged)
################################################

################################################
# Page: Pipeline and Model Training (Merged)
################################################

elif page == "Pipeline and Model Training":
    st.header("Pipeline and Model Training")
    ensure_session_state()
    inject_css()
    init_groq_client()

    st.markdown("# Pipeline and Model Training")
    st.caption("Build preprocessing pipelines, select models, and configure hyperparameter tuning strategies")

    if st.session_state.df is None:
        st.warning("No dataset loaded. Please upload and split data first.")
        st.stop()

    # Get splits
    splits = st.session_state.get("split_result")
    if not splits:
        st.warning("Please perform Train-Test Split first.")
        st.stop()

    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    cols = X_train.columns.tolist()

    # --- Data Validation Section ---
    st.markdown("## Data Validation")

    original_df = st.session_state.df
    all_columns = original_df.columns.tolist()

    if 'target_column' not in st.session_state:
        target_col = st.selectbox("Select target column", [None] + all_columns)
        if target_col:
            st.session_state.target_column = target_col
        else:
            st.info("Please select a target column to continue.")
            st.stop()
    else:
        target_col = st.session_state.target_column

    # Check if target variable is accidentally in features
    if target_col in X_train.columns:
        st.error(f"❌ CRITICAL ERROR: Target variable '{target_col}' is in the feature columns!")
        st.error("This will cause perfect accuracy (1.0) because the model can see the answers.")
        st.stop()

    # Check for single class in target
    if y_train.nunique() == 1:
        st.error(f"Only one class found in target variable: {y_train.unique()[0]}")
        st.error("This will always result in perfect accuracy for that class.")
        st.stop()

    # --- Column assignment ---
    st.markdown("## Assign Columns")
    num_cols = st.multiselect("Numeric Columns", cols, default=X_train.select_dtypes(include=np.number).columns.tolist())
    cat_cols = st.multiselect("Categorical Columns", [c for c in cols if c not in num_cols], default=X_train.select_dtypes(exclude=np.number).columns.tolist())

    st.info(f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

    # --- Numeric Pipeline Builder ---
    st.markdown("## Numeric Pipeline")
    num_imputer = st.selectbox("Imputer", ["Mean", "Median", "Most Frequent", "Constant", "KNN", "Drop Rows", "None"])
    num_scaler = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])

    num_steps = []
    if num_imputer == "Mean":
        num_steps.append(("imputer", SimpleImputer(strategy="mean")))
    elif num_imputer == "Median":
        num_steps.append(("imputer", SimpleImputer(strategy="median")))
    elif num_imputer == "Most Frequent":
        num_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    elif num_imputer == "Constant":
        num_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0)))
    elif num_imputer == "KNN":
        num_steps.append(("imputer", KNNImputer(n_neighbors=5)))
    elif num_imputer == "Drop Rows":
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        X_test = X_test.dropna()
        y_test = y_test.loc[X_test.index]

    if num_scaler == "StandardScaler":
        num_steps.append(("scaler", StandardScaler()))
    elif num_scaler == "MinMaxScaler":
        num_steps.append(("scaler", MinMaxScaler()))
    elif num_scaler == "RobustScaler":
        num_steps.append(("scaler", RobustScaler()))

    num_pipeline = Pipeline(num_steps) if num_steps else "passthrough"

    # --- Categorical Pipeline Builder ---
    st.markdown("## Categorical Pipeline")
    cat_imputer = st.selectbox("Imputer", ["Most Frequent", "Constant", "None"])
    cat_encoder = st.selectbox("Encoder", ["Ordinal", "OneHot", "None"])

    cat_steps = []
    if cat_imputer == "Most Frequent":
        cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    elif cat_imputer == "Constant":
        cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="missing")))

    if cat_encoder == "Ordinal":
        cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
    elif cat_encoder == "OneHot":
        cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    elif cat_encoder == "None":
        st.warning("Categorical features must be numeric for most models. Falling back to OrdinalEncoder.")
        cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

    cat_pipeline = Pipeline(cat_steps) if cat_steps else "passthrough"

    # --- Build final preprocessor ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="passthrough"
    )

    st.success("Preprocessor ready. Now choose models below.")

    # --- Model Selection ---
    st.markdown("## Model Selection")

    def detect_task_type(y):
        """Better task type detection"""
        if pd.api.types.is_numeric_dtype(y):
            unique_values = y.nunique()
            if unique_values <= 15:
                unique_vals = sorted(y.dropna().unique())
                if (all(isinstance(v, (int, np.integer)) for v in unique_vals) and
                    all(v in range(len(unique_vals)) for v in unique_vals)):
                    return "classification"
                else:
                    return "regression"
            else:
                return "regression"
        else:
            return "classification"

    task_type = detect_task_type(y_train)
    st.info(f"Detected Task Type: **{task_type}**")

    # Handle label encoding for XGBoost and LightGBM if needed
    y_train_encoded = y_train.copy()
    y_test_encoded = y_test.copy()
    label_encoder = None
    
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y_train):
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        st.info("Target labels encoded for compatibility with XGBoost/LightGBM")

    # Model options
    models = {}

    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False),
            "LightGBM": lgb.LGBMClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "Neural Network (MLP) - Small": MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=1000),
            "Neural Network (MLP) - Medium": MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            "Neural Network (MLP) - Large": MLPClassifier(hidden_layer_sizes=(100, 100, 50), random_state=42, max_iter=1000)
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=42),
            "Lasso": Lasso(alpha=1.0, random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVR": SVR(C=1.0, kernel='rbf'),
            "XGBoost": xgb.XGBRegressor(random_state=42),
            "LightGBM": lgb.LGBMRegressor(random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42),
            "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
            "Neural Network (MLP) - Small": MLPRegressor(hidden_layer_sizes=(50,), random_state=42, max_iter=1000),
            "Neural Network (MLP) - Medium": MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            "Neural Network (MLP) - Large": MLPRegressor(hidden_layer_sizes=(100, 100, 50), random_state=42, max_iter=1000)
        }

    # Change from selectbox to multiselect
    selected_model_names = st.multiselect(
        "Select Model(s)", 
        list(models.keys()),
        default=[list(models.keys())[0]] if models else []
    )

    selected_models = {name: models[name] for name in selected_model_names}

    # --- Hyperparameter Tuning Strategy ---
    st.markdown("## Hyperparameter Tuning Strategy")
    
    # Initialize session state for tuning strategy
    if 'tuning_strategy' not in st.session_state:
        st.session_state.tuning_strategy = "default"
    if 'tuning_results' not in st.session_state:
        st.session_state.tuning_results = {}
    
    tuning_strategy = st.radio(
        "Tuning Strategy",
        ["Use Default Parameters", "Use Automated Tuning"],
        key="tuning_strategy_radio"
    )
    
    # Map radio selection to strategy values
    if tuning_strategy == "Use Default Parameters":
        strategy_value = "default"
    else:
        strategy_value = "tune"
    
    st.session_state.tuning_strategy = strategy_value
    
    # Default parameter values (single values, not lists)
    default_param_values = {
        "Logistic Regression": {
            'model__C': 1.0,
            'model__penalty': 'l2',
            'model__solver': 'lbfgs'
        },
        "Random Forest": {
            'model__n_estimators': 100,
            'model__max_depth': None,
            'model__min_samples_split': 2
        },
        "Gradient Boosting": {
            'model__n_estimators': 100,
            'model__learning_rate': 0.1,
            'model__max_depth': 3
        },
        "SVM": {
            'model__C': 1.0,
            'model__kernel': 'rbf',
            'model__gamma': 'scale'
        },
        "XGBoost": {
            'model__n_estimators': 100,
            'model__max_depth': 3,
            'model__learning_rate': 0.1
        },
        "LightGBM": {
            'model__n_estimators': 100,
            'model__learning_rate': 0.1,
            'model__max_depth': 3
        }
    }
    
    # Parameter grids for tuning (lists of values)
    default_param_grids = {
        "Logistic Regression": {
            'model__C': [0.1, 1, 10],
            'model__penalty': ['l2', 'none'],
            'model__solver': ['lbfgs', 'saga']
        },
        "Random Forest": {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_split': [2, 5, 10]
        },
        "Gradient Boosting": {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        },
        "SVM": {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf'],
            'model__gamma': ['scale', 'auto']
        },
        "XGBoost": {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2]
        },
        "LightGBM": {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        }
    }
    
    # Automated Tuning Section - Only show if models are selected
    if strategy_value == "tune" and selected_model_names:
        st.markdown("### Automated Tuning Techniques")
        
        # Let user select which model to tune
        model_to_tune = st.selectbox(
            "Select model to tune",
            options=selected_model_names,
            key="model_tune_select"
        )
        
        tuning_method = st.selectbox(
            "Tuning Method",
            ["GridSearchCV", "RandomizedSearchCV", "Bayesian Optimization (Optuna)"],
            key="tuning_method"
        )
        
        cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5, key="cv_folds")
        
        # AI-powered tuning recommendations
        if st.session_state.groq_available:
            with st.expander("🤖 AI Tuning Recommendations"):
                if st.button("Get AI Tuning Advice", key="ai_tuning_advice"):
                    context = f"""
                    Dataset: {st.session_state.get('dataset_name', 'Unknown')}
                    Shape: {X_train.shape}
                    Task Type: {task_type}
                    Model: {model_to_tune}
                    Tuning Method: {tuning_method}
                    """
                    
                    prompt = f"""
                    You are a machine learning expert. Provide specific advice for hyperparameter tuning.
                    
                    Context:
                    {context}
                    
                    Please provide:
                    1. Recommended parameter ranges for {model_to_tune}
                    2. Best tuning strategy for this model type
                    3. Expected performance improvement from tuning
                    4. Any warnings or considerations
                    
                    Keep the response concise and practical.
                    """
                    
                    with st.spinner("Getting AI recommendations..."):
                        response = call_groq_llm(prompt)
                        st.info(response)
        
        if st.button("Run Tuning", key="run_tuning"):
            with st.spinner(f"Running {tuning_method} for {model_to_tune}..."):
                try:
                    # Get the selected model
                    selected_model = models[model_to_tune]
                    default_params = default_param_grids.get(model_to_tune, {})
                    
                    # Create pipeline
                    pipe = Pipeline([
                        ("preprocessor", preprocessor),
                        ("model", selected_model)
                    ])
                    
                    if tuning_method == "GridSearchCV":
                        search = GridSearchCV(
                            pipe, 
                            default_params, 
                            cv=cv_folds,
                            scoring='accuracy' if task_type == 'classification' else 'r2',
                            n_jobs=-1,
                            return_train_score=True
                        )
                        # Use encoded labels for XGBoost/LightGBM if needed
                        if model_to_tune in ["XGBoost", "LightGBM"] and label_encoder is not None:
                            search.fit(X_train, y_train_encoded)
                        else:
                            search.fit(X_train, y_train)
                        
                    elif tuning_method == "RandomizedSearchCV":
                        search = RandomizedSearchCV(
                            pipe,
                            default_params,
                            n_iter=10,
                            cv=cv_folds,
                            scoring='accuracy' if task_type == 'classification' else 'r2',
                            n_jobs=-1,
                            random_state=42,
                            return_train_score=True
                        )
                        # Use encoded labels for XGBoost/LightGBM if needed
                        if model_to_tune in ["XGBoost", "LightGBM"] and label_encoder is not None:
                            search.fit(X_train, y_train_encoded)
                        else:
                            search.fit(X_train, y_train)
                        
                    else:  # Bayesian Optimization with Optuna
                        def objective(trial):
                            # Define parameter space for Optuna
                            params = {}
                            for param_name, param_values in default_params.items():
                                clean_name = param_name.replace('model__', '')
                                
                                if isinstance(param_values[0], (int, np.integer)):
                                    params[param_name] = trial.suggest_int(clean_name, min(param_values), max(param_values))
                                elif isinstance(param_values[0], float):
                                    params[param_name] = trial.suggest_float(clean_name, min(param_values), max(param_values))
                                else:
                                    params[param_name] = trial.suggest_categorical(clean_name, param_values)
                            
                            # Create model with suggested parameters
                            model_clone = clone(selected_model)
                            model_clone.set_params(**{k.replace('model__', ''): v for k, v in params.items()})
                            
                            pipe = Pipeline([
                                ("preprocessor", preprocessor),
                                ("model", model_clone)
                            ])
                            
                            # Use encoded labels for XGBoost/LightGBM if needed
                            if model_to_tune in ["XGBoost", "LightGBM"] and label_encoder is not None:
                                return cross_val_score(pipe, X_train, y_train_encoded, cv=cv_folds, 
                                                     scoring='accuracy' if task_type == 'classification' else 'r2').mean()
                            else:
                                return cross_val_score(pipe, X_train, y_train, cv=cv_folds, 
                                                     scoring='accuracy' if task_type == 'classification' else 'r2').mean()
                        
                        study = optuna.create_study(direction="maximize")
                        study.optimize(objective, n_trials=20)
                        
                        # Convert Optuna results to match sklearn format
                        search = type('obj', (object,), {
                            'best_params_': study.best_params,
                            'best_score_': study.best_value,
                            'cv_results_': None
                        })()
                    
                    # Store results
                    st.session_state.tuning_results[model_to_tune] = {
                        'best_params': search.best_params_,
                        'best_score': search.best_score_,
                        'method': tuning_method
                    }
                    
                    st.success("Tuning completed successfully!")
                    
                    # Display tuning results
                    st.markdown("#### Tuning Results")
                    st.json(search.best_params_)
                    st.write(f"Best Score: {search.best_score_:.4f}")
                    st.write(f"Method: {tuning_method}")
                    
                    # Visualize optimization history for Optuna
                    if tuning_method == "Bayesian Optimization (Optuna)":
                        try:
                            fig = plot_optimization_history(study)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                            
                except Exception as e:
                    st.error(f"Tuning failed: {str(e)}")
    
    # --- AI Assistant Integration ---
    if st.session_state.groq_available:
        with st.expander("✅ AI Assistant", expanded=False):
            st.markdown("Get expert advice on model selection and hyperparameter tuning")
            
            # Context information for the AI
            context = f"""
            Dataset: {st.session_state.get('dataset_name', 'Unknown')}
            Shape: {X_train.shape}
            Task Type: {task_type}
            Tuning Strategy: {tuning_strategy}
            Numeric Features: {num_cols}
            Categorical Features: {cat_cols}
            Selected Models: {selected_model_names}
            """
            
            question = st.text_input(
                "Ask the AI assistant about model selection or tuning:",
                placeholder="e.g., 'Which model is best for my data?', 'How should I tune learning rate?'",
                key="ai_question"
            )
            
            if st.button("Get AI Advice", key="ai_advice"):
                if question:
                    with st.spinner("Consulting AI expert..."):
                        prompt = f"""
                        You are a machine learning expert. Provide specific, actionable advice.
                        
                        Context:
                        {context}
                        
                        Question: {question}
                        
                        Please provide:
                        1. Specific recommendations based on the dataset characteristics
                        2. Explanation of why these choices might work well
                        3. Any warnings or considerations
                        4. Suggested next steps
                        
                        Keep the response concise and practical.
                        """
                        
                        response = call_groq_llm(prompt)
                        st.info(response)
                else:
                    st.warning("Please enter a question first.")
    
    # --- Training Execution ---
    st.markdown("## Training Control")

    # Determine which parameters to use for each model
    models_params = {}
    for model_name in selected_model_names:
        if strategy_value == "default":
            models_params[model_name] = default_param_values.get(model_name, {})
        elif strategy_value == "tune" and model_name in st.session_state.tuning_results:
            # Use the best parameters from tuning
            models_params[model_name] = st.session_state.tuning_results[model_name]['best_params']
        else:
            models_params[model_name] = {}

    if st.button("Train Selected Models", key="train_model_final", type="primary"):
        if not selected_model_names:
            st.warning("Please select at least one model to train.")
        else:
            # Initialize session state for trained models
            if 'trained_models' not in st.session_state:
                st.session_state.trained_models = {}
            if 'model_results' not in st.session_state:
                st.session_state.model_results = {}
            
            # Clear previous results
            st.session_state.trained_models = {}
            st.session_state.model_results = {}
            
            for model_name in selected_model_names:
                try:
                    model = models[model_name]
                    params_to_use = models_params[model_name]
                    
                    # Create pipeline
                    pipe = Pipeline([
                        ("preprocessor", preprocessor),
                        ("model", model)
                    ])
                    
                    # Set parameters if any are specified
                    if params_to_use:
                        pipe.set_params(**params_to_use)
                    
                    # Train the model
                    with st.spinner(f"Training {model_name}..."):
                        start_time = time.time()
                        
                        # Use encoded labels for XGBoost/LightGBM if needed
                        if model_name in ["XGBoost", "LightGBM"] and label_encoder is not None:
                            pipe.fit(X_train, y_train_encoded)
                        else:
                            pipe.fit(X_train, y_train)
                            
                        training_time = time.time() - start_time
                    
                    # Make predictions
                    if model_name in ["XGBoost", "LightGBM"] and label_encoder is not None:
                        y_pred = pipe.predict(X_test)
                        # Decode predictions back to original labels
                        y_pred_decoded = label_encoder.inverse_transform(y_pred)
                        y_test_for_eval = y_test
                    else:
                        y_pred = pipe.predict(X_test)
                        y_pred_decoded = y_pred
                        y_test_for_eval = y_test
                    
                    # Calculate metrics
                    if task_type == "classification":
                        acc = accuracy_score(y_test_for_eval, y_pred_decoded)
                        f1 = f1_score(y_test_for_eval, y_pred_decoded, average="weighted")
                        prec = precision_score(y_test_for_eval, y_pred_decoded, average="weighted", zero_division=0)
                        rec = recall_score(y_test_for_eval, y_pred_decoded, average="weighted", zero_division=0)
                        
                        metrics = {
                            "Accuracy": acc, 
                            "Precision": prec, 
                            "Recall": rec, 
                            "F1": f1,
                            "Training Time (s)": training_time
                        }
                    else:
                        r2 = r2_score(y_test_for_eval, y_pred_decoded)
                        mae = mean_absolute_error(y_test_for_eval, y_pred_decoded)
                        mse = mean_squared_error(y_test_for_eval, y_pred_decoded)
                        rmse = np.sqrt(mse)
                        
                        metrics = {
                            "R²": r2, 
                            "MAE": mae, 
                            "MSE": mse, 
                            "RMSE": rmse,
                            "Training Time (s)": training_time
                        }
                    
                    # Store results
                    st.session_state.trained_models[model_name] = pipe
                    st.session_state.model_results[model_name] = metrics
                    
                    st.success(f"{model_name} trained successfully!")
                    
                except Exception as e:
                    st.error(f"Training failed for {model_name}: {str(e)}")
            
            # Determine best model after training all
            if 'model_results' in st.session_state and st.session_state.model_results:
                best_model_name = None
                best_metric_value = -float('inf') if task_type == "classification" else float('inf')
                
                for model_name, metrics in st.session_state.model_results.items():
                    # Use the first metric for comparison
                    primary_metric = next(iter(metrics.values()))
                    if (task_type == "classification" and primary_metric > best_metric_value) or \
                    (task_type == "regression" and primary_metric < best_metric_value):
                        best_metric_value = primary_metric
                        best_model_name = model_name
                
                if best_model_name:
                    st.session_state.best_model = {
                        "name": best_model_name,
                        "pipeline": st.session_state.trained_models[best_model_name],
                        "metrics": st.session_state.model_results[best_model_name],
                        "label_encoder": label_encoder
                    }
                    st.success(f"🎯 Best model: {best_model_name}")
                    
                    # Save best model to disk for Final Evaluation page
                    try:
                        best_model_data = {
                            "pipeline": st.session_state.best_model["pipeline"],
                            "metrics": st.session_state.best_model["metrics"],
                            "model_name": st.session_state.best_model["name"],
                            "training_date": datetime.now().isoformat(),
                            "task_type": task_type,
                            "label_encoder": label_encoder
                        }
                        
                        joblib.dump(best_model_data, "best_model_pipeline.joblib")
                        st.info("Best model saved for evaluation on Final Evaluation page")
                    except Exception as e:
                        st.warning(f"Could not save best model: {e}")
    
    # --- Model Comparison ---
    if 'model_results' in st.session_state and st.session_state.model_results:
        st.markdown("## Model Comparison")
        
        comparison_data = []
        for model_name, metrics in st.session_state.model_results.items():
            row = {"Model": model_name}
            row.update(metrics)
            if 'best_model' in st.session_state and st.session_state.best_model["name"] == model_name:
                row["Best"] = "⭐"
            else:
                row["Best"] = ""
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Style the comparison table
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # For classification, highlight max for all metrics except training time
            if task_type == "classification":
                styled_df = comparison_df.style.highlight_max(
                    subset=[col for col in numeric_cols if col != "Training Time (s)"], 
                    color='lightgreen'
                ).highlight_min(
                    subset=["Training Time (s)"], 
                    color='lightgreen'
                )
            # For regression, highlight min for error metrics and max for R²
            else:
                # Highlight max for R²
                if "R²" in comparison_df.columns:
                    styled_df = comparison_df.style.highlight_max(
                        subset=["R²"], 
                        color='lightgreen'
                    )
                # Highlight min for error metrics
                error_metrics = [col for col in numeric_cols if col in ["MAE", "MSE", "RMSE"]]
                if error_metrics:
                    styled_df = styled_df.highlight_min(
                        subset=error_metrics, 
                        color='lightgreen'
                    ).highlight_min(
                        subset=["Training Time (s)"], 
                        color='lightgreen'
                    )
            
            st.dataframe(styled_df, use_container_width=True)
            
        # Link to Final Evaluation page
        st.markdown("---")
        st.success("✅ Models trained successfully!")
        st.markdown("### Next Steps")
        st.markdown("Proceed to the **Final Evaluation** page to:")
        st.markdown("- Analyze model performance in detail")
        st.markdown("- View confusion matrices and ROC curves")
        st.markdown("- Generate SHAP explanations")
        st.markdown("- Compare all trained models")
        st.markdown("- Download comprehensive evaluation reports")
        
        if st.button("Go to Final Evaluation", key="go_to_evaluation"):
            # Set the page to Final Evaluation programmatically
            st.session_state.page = "Final Evaluation"
            st.rerun()

#########################################
# Page 5: Final Evaluation
#########################################
elif page == "Final Evaluation":
    st.header("Final Evaluation")

    ensure_session_state()
    inject_css()

    st.markdown("# Final Evaluation")
    st.caption("Comprehensive model evaluation with industry metrics, compact visualizations, and explainable AI insights.")

    if st.session_state.df is None:
        st.warning("No dataset loaded. Please upload and split data first.")
        st.stop()

    splits = st.session_state.get("split_result")
    if not splits:
        st.warning("Please perform Train-Test Split and build pipeline first.")
        st.stop()

    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    # Load trained models for comparison
    models_to_compare = {}
    for name in ["best_model_pipeline.joblib", "alt_model_pipeline.joblib"]:
        try:
            loaded_obj = joblib.load(name)
            # Extract pipeline from dictionary if needed
            if isinstance(loaded_obj, dict) and 'pipeline' in loaded_obj:
                models_to_compare[name] = loaded_obj['pipeline']
            else:
                models_to_compare[name] = loaded_obj
        except Exception:
            continue

    if not models_to_compare:
        st.warning("No trained models found. Please train and save models first.")
        st.stop()

    # Detect task type
    task_type = "classification" if len(np.unique(y_train)) < 20 and y_train.dtype != float else "regression"
    
    # Create tabs for different evaluation aspects
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance Metrics", 
        "📈 Visualizations", 
        "🔍 Model Explainability", 
        "📋 Comparison Report",
        "💾 Download Results"
    ])

    results_summary = {}
    detailed_results = {}

    with tab1:
        st.subheader("Model Performance Metrics")
        
        # Create columns for model comparison
        cols = st.columns(len(models_to_compare))
        
        for i, (label, model) in enumerate(models_to_compare.items()):
            with cols[i]:
                st.markdown(f"**{label.replace('_pipeline.joblib', '').replace('_', ' ').title()}**")
                
                start = time.time()
                
                # Check if the loaded object is a dictionary containing a pipeline
            if isinstance(model, dict) and 'pipeline' in model:
                y_pred = model['pipeline'].predict(X_test)
            else:
                # If it's already a pipeline object
                y_pred = model.predict(X_test)

            if isinstance(model, dict) and 'pipeline' in model:
                y_prob = model['pipeline'].predict(X_test) if hasattr(model, "predict_proba") else None
            else:
                y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                runtime = time.time() - start
                
                # Store for later use
                detailed_results[label] = {
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                    "runtime": runtime
                }
                
                if task_type == "classification":
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    
                    # Industry-specific metrics
                    if len(np.unique(y_test)) == 2:  # Binary classification
                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        
                        metrics = {
                            "Accuracy": f"{acc:.3f}",
                            "Precision": f"{prec:.3f}",
                            "Recall (Sensitivity)": f"{rec:.3f}",
                            "Specificity": f"{specificity:.3f}",
                            "F1 Score": f"{f1:.3f}",
                            "NPV": f"{npv:.3f}",
                            "FPR": f"{fpr:.3f}",
                            "Runtime (s)": f"{runtime:.3f}"
                        }
                    else:
                        metrics = {
                            "Accuracy": f"{acc:.3f}",
                            "Precision": f"{prec:.3f}",
                            "Recall": f"{rec:.3f}",
                            "F1 Score": f"{f1:.3f}",
                            "Runtime (s)": f"{runtime:.3f}"
                        }
                    
                    # Display metrics in a compact way
                    for metric, value in metrics.items():
                        st.metric(metric, value)
                        
                    results_summary[label] = metrics
                    
                else:  # Regression
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    # Industry-specific metrics
                    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1))) * 100  # Avoid division by zero
                    wmape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100
                    
                    metrics = {
                        "R²": f"{r2:.3f}",
                        "MAE": f"{mae:.3f}",
                        "RMSE": f"{rmse:.3f}",
                        "MAPE": f"{mape:.1f}%",
                        "WMAPE": f"{wmape:.1f}%",
                        "Runtime (s)": f"{runtime:.3f}"
                    }
                    
                    # Display metrics in a compact way
                    for metric, value in metrics.items():
                        st.metric(metric, value)
                        
                    results_summary[label] = metrics

    with tab2:
        st.subheader("Model Performance Visualizations")
        
        model_choice = st.selectbox("Select model to visualize", list(models_to_compare.keys()), key="viz_model")
        model = models_to_compare[model_choice]
        y_pred = detailed_results[model_choice]["y_pred"]
        y_prob = detailed_results[model_choice]["y_prob"]
        
        if task_type == "classification":
            # Create a 2x2 grid of small plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Compact confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
                
                # Precision-Recall curve for binary classification
                if y_prob is not None and len(np.unique(y_test)) == 2:
                    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob[:, 1])
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.plot(rec_curve, prec_curve)
                    ax.set_xlabel("Recall")
                    ax.set_ylabel("Precision")
                    ax.set_title("Precision-Recall Curve")
                    st.pyplot(fig)
            
            with col2:
                # ROC curve for binary classification
                if y_prob is not None and len(np.unique(y_test)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                    auc_score = roc_auc_score(y_test, y_prob[:, 1])
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
                    ax.plot([0, 1], [0, 1], linestyle="--")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("ROC Curve")
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                
                # Feature importance if available
                try:
                    if hasattr(model.named_steps["model"], "feature_importances_"):
                        importances = model.named_steps["model"].feature_importances_
                        # Get feature names after preprocessing
                        preprocessor = model.named_steps["preprocessor"]
                        try:
                            feature_names = preprocessor.get_feature_names_out()
                        except:
                            feature_names = [f"feature_{i}" for i in range(len(importances))]
                        
                        # Plot top 10 features
                        indices = np.argsort(importances)[-10:]
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.barh(range(len(indices)), importances[indices])
                        ax.set_yticks(range(len(indices)))
                        ax.set_yticklabels([feature_names[i] for i in indices])
                        ax.set_title("Top 10 Feature Importances")
                        st.pyplot(fig)
                except:
                    pass
                    
        else:  # Regression
            col1, col2 = st.columns(2)
            
            with col1:
                # Residual plot
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.scatter(y_pred, residuals, alpha=0.5, s=10)
                ax.axhline(0, linestyle="--", color="red")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Residuals")
                ax.set_title("Residual Plot")
                st.pyplot(fig)
            
            with col2:
                # Prediction vs Actual plot
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.scatter(y_test, y_pred, alpha=0.5, s=10)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Predicted vs Actual")
                st.pyplot(fig)
                
            # Error distribution
            errors = y_test - y_pred
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.hist(errors, bins=30, alpha=0.7)
            ax.axvline(0, color='r', linestyle='--')
            ax.set_xlabel("Prediction Error")
            ax.set_ylabel("Frequency")
            ax.set_title("Error Distribution")
            st.pyplot(fig)

    with tab3:
        st.subheader("Model Explainability with SHAP")
        
        if len(models_to_compare) > 0:
            selected_model_name = st.selectbox(
                "Select model for SHAP analysis",
                options=list(models_to_compare.keys()),
                key="shap_model_select"
            )
            
            selected_model = models_to_compare[selected_model_name]
            
            try:
                # Check if model supports SHAP
                model_for_shap = selected_model.named_steps["model"]
                
                # Preprocess the test data using the pipeline's preprocessor
                preprocessor = selected_model.named_steps["preprocessor"]
                X_test_processed = preprocessor.transform(X_test)
                
                # Get feature names after preprocessing
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    # Fallback for older sklearn versions
                    feature_names = [f"feature_{i}" for i in range(X_test_processed.shape[1])]
                
                # Create explainer - handle different model types
                try:
                    # Try TreeExplainer first for tree-based models
                    if isinstance(model_for_shap, (RandomForestClassifier, RandomForestRegressor, 
                                                GradientBoostingClassifier, GradientBoostingRegressor,
                                                xgb.XGBClassifier, xgb.XGBRegressor,
                                                lgb.LGBMClassifier, lgb.LGBMRegressor,
                                                DecisionTreeClassifier, DecisionTreeRegressor)):
                        explainer = shap.TreeExplainer(model_for_shap)
                        shap_values = explainer.shap_values(X_test_processed)
                    else:
                        # Use KernelExplainer for other models
                        explainer = shap.KernelExplainer(model_for_shap.predict, shap.sample(X_test_processed, 100))  # Sample for speed
                        shap_values = explainer.shap_values(shap.sample(X_test_processed, 50))  # Smaller sample
                except:
                    # Fallback to LinearExplainer
                    explainer = shap.LinearExplainer(model_for_shap, X_test_processed)
                    shap_values = explainer.shap_values(X_test_processed)
                
                st.success("✅ SHAP analysis successful!")
                
                # SHAP Summary Plot (compact version)
                st.markdown("#### Feature Importance Summary")
                try:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        # Multi-class classification - show mean absolute SHAP values
                        shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, 
                                         plot_type="bar", max_display=10, show=False)
                    else:
                        # Binary classification or regression
                        shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, 
                                         max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning(f"Summary plot failed: {e}")
                
                # SHAP Dependence Plot
                st.markdown("#### Feature Effects Analysis")
                feature_for_dependence = st.selectbox("Select feature for detailed analysis", 
                                                options=feature_names, key="shap_dependence_feature")
                
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    if isinstance(shap_values, list):
                        shap.dependence_plot(feature_for_dependence, shap_values[0], X_test_processed, 
                                        feature_names=feature_names, ax=ax, show=False)
                    else:
                        shap.dependence_plot(feature_for_dependence, shap_values, X_test_processed, 
                                        feature_names=feature_names, ax=ax, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning(f"Dependence plot failed: {e}")
                    
                # SHAP Force Plot for a single instance
                st.markdown("#### Individual Prediction Explanation")
                instance_idx = st.slider("Select instance to explain", 0, min(20, len(X_test_processed)-1), 0)
                
                try:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    if isinstance(shap_values, list):
                        # For multi-class, show explanation for first class
                        shap.force_plot(explainer.expected_value[0], shap_values[0][instance_idx], 
                                    X_test_processed[instance_idx], feature_names=feature_names, 
                                    matplotlib=True, show=False)
                    else:
                        shap.force_plot(explainer.expected_value, shap_values[instance_idx], 
                                    X_test_processed[instance_idx], feature_names=feature_names, 
                                    matplotlib=True, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show the actual instance values
                    st.markdown("**Instance feature values:**")
                    instance_data = {}
                    for i, name in enumerate(feature_names):
                        instance_data[name] = X_test_processed[instance_idx, i]
                    st.write(pd.DataFrame.from_dict(instance_data, orient='index', columns=['Value']).head(10))
                    
                except Exception as e:
                    st.warning(f"Force plot failed: {e}")
                    
            except Exception as e:
                st.error(f"SHAP analysis failed: {str(e)}")
                st.info("""
                **Common SHAP issues:**
                - Model type not supported by SHAP
                - Large datasets may timeout
                - Preprocessing issues
                - Try a tree-based model (Random Forest, XGBoost) for better SHAP support
                """)
        else:
            st.info("No trained models available for SHAP analysis")

    with tab4:
        st.subheader("Model Comparison Report")
        
        # Create a comprehensive comparison table
        comparison_df = pd.DataFrame(results_summary).T
        st.dataframe(comparison_df.style.highlight_max(axis=0, color='#90EE90').highlight_min(axis=0, color='#FFCCCB'))
        
        # Add statistical tests for model comparison
        if len(models_to_compare) > 1:
            st.markdown("#### Statistical Significance Testing")
            
            if task_type == "classification":
                # McNemar's test for classifier comparison
                from statsmodels.stats.contingency_tables import mcnemar
                
                model_names = list(models_to_compare.keys())
                y_preds = [detailed_results[name]["y_pred"] for name in model_names]
                
                # Create a matrix of p-values
                p_values = np.ones((len(model_names), len(model_names)))
                
                for i in range(len(model_names)):
                    for j in range(i+1, len(model_names)):
                        # Create contingency table
                        correct_i = (y_preds[i] == y_test)
                        correct_j = (y_preds[j] == y_test)
                        
                        both_correct = np.sum(correct_i & correct_j)
                        both_wrong = np.sum((~correct_i) & (~correct_j))
                        i_correct_j_wrong = np.sum(correct_i & (~correct_j))
                        i_wrong_j_correct = np.sum((~correct_i) & correct_j)
                        
                        table = [[both_correct, i_correct_j_wrong],
                                [i_wrong_j_correct, both_wrong]]
                        
                        # Perform McNemar's test
                        result = mcnemar(table, exact=False)
                        p_values[i, j] = result.pvalue
                        p_values[j, i] = result.pvalue
                
                # Display results
                p_df = pd.DataFrame(p_values, index=model_names, columns=model_names)
                st.write("McNemar's test p-values (lower values indicate significant differences):")
                st.dataframe(p_df.style.format("{:.4f}").applymap(
                    lambda x: 'background-color: yellow' if x < 0.05 else ''))
                
            else:
                # Paired t-test for regression models
                from scipy.stats import ttest_rel
                
                model_names = list(models_to_compare.keys())
                y_preds = [detailed_results[name]["y_pred"] for name in model_names]
                errors = [np.abs(y_test - pred) for pred in y_preds]
                
                # Create a matrix of p-values
                p_values = np.ones((len(model_names), len(model_names)))
                
                for i in range(len(model_names)):
                    for j in range(i+1, len(model_names)):
                        # Perform paired t-test
                        t_stat, p_val = ttest_rel(errors[i], errors[j])
                        p_values[i, j] = p_val
                        p_values[j, i] = p_val
                
                # Display results
                p_df = pd.DataFrame(p_values, index=model_names, columns=model_names)
                st.write("Paired t-test p-values for absolute errors (lower values indicate significant differences):")
                st.dataframe(p_df.style.format("{:.4f}").applymap(
                    lambda x: 'background-color: yellow' if x < 0.05 else ''))
        
        # Business impact analysis
        st.markdown("#### Business Impact Analysis")
        
        if task_type == "classification" and len(np.unique(y_test)) == 2:
            # For binary classification, calculate business metrics
            st.info("Assuming a business context where:")
            col1, col2 = st.columns(2)
            
            with col1:
                tp_value = st.number_input("True Positive Value ($)", value=1000, key="tp_value")
                fp_cost = st.number_input("False Positive Cost ($)", value=500, key="fp_cost")
            
            with col2:
                fn_cost = st.number_input("False Negative Cost ($)", value=2000, key="fn_cost")
                tn_value = st.number_input("True Negative Value ($)", value=100, key="tn_value")
            
            business_results = {}
            for label, model in models_to_compare.items():
                y_pred = detailed_results[label]["y_pred"]
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                
                profit = (tp * tp_value) + (tn * tn_value) - (fp * fp_cost) - (fn * fn_cost)
                roi = profit / (len(y_test) * (fp_cost + fn_cost + tp_value + tn_value) / 4) * 100
                
                business_results[label] = {
                    "Profit ($)": profit,
                    "ROI (%)": roi,
                    "Cost Avoidance ($)": (fn * fn_cost) + (fp * fp_cost)
                }
            
            business_df = pd.DataFrame(business_results).T
            st.dataframe(business_df.style.highlight_max(axis=0, color='#90EE90'))
            
        elif task_type == "regression":
            # For regression, calculate cost of error
            st.info("Assuming a business context where prediction errors have costs:")
            error_cost_per_unit = st.number_input("Cost per unit of error ($)", value=50, key="error_cost")
            
            business_results = {}
            for label, model in models_to_compare.items():
                y_pred = detailed_results[label]["y_pred"]
                mae = mean_absolute_error(y_test, y_pred)
                total_cost = mae * len(y_test) * error_cost_per_unit
                
                business_results[label] = {
                    "Total Error Cost ($)": total_cost,
                    "Cost per Prediction ($)": total_cost / len(y_test)
                }
            
            business_df = pd.DataFrame(business_results).T
            st.dataframe(business_df.style.highlight_min(axis=0, color='#90EE90'))

    with tab5:
        st.subheader("Download Evaluation Results")
        
        # Create a comprehensive report
        report_data = {
            "Model Comparison": pd.DataFrame(results_summary).T,
            "Dataset Info": {
                "Samples": len(X_test),
                "Features": X_test.shape[1],
                "Task Type": task_type
            }
        }
        
        # Convert to JSON for download
        report_json = json.dumps({
            "model_comparison": pd.DataFrame(results_summary).T.to_dict(),
            "dataset_info": report_data["Dataset Info"],
            "timestamp": datetime.now().isoformat()
        }, indent=2)
        
        st.download_button(
            "Download Full Report (JSON)",
            data=report_json,
            file_name="model_evaluation_report.json",
            mime="application/json"
        )
        
        # Download SHAP values if available
        if 'shap_values' in locals():
            shap_df = pd.DataFrame(shap_values, columns=feature_names)
            csv = shap_df.to_csv(index=False)
            st.download_button(
                "Download SHAP Values (CSV)",
                data=csv,
                file_name="shap_values.csv",
                mime="text/csv"
            )
        
        # Download predictions
        predictions_df = pd.DataFrame({
            "Actual": y_test,
            **{f"Predicted_{label}": detailed_results[label]["y_pred"] for label in models_to_compare.keys()}
        })
        
        st.download_button(
            "Download Predictions (CSV)",
            data=predictions_df.to_csv(index=False),
            file_name="model_predictions.csv",
            mime="text/csv"
        )

    # Add a summary at the bottom
    st.markdown("---")
    st.subheader("Evaluation Summary")
    
    # Find best model based on primary metric
    if task_type == "classification":
        best_model = max(results_summary, key=lambda x: float(results_summary[x]['Accuracy']))
        st.success(f"**Best Model:** {best_model.replace('_pipeline.joblib', '')} "
                  f"(Accuracy: {results_summary[best_model]['Accuracy']})")
    else:
        best_model = min(results_summary, key=lambda x: float(results_summary[x]['RMSE']))
        st.success(f"**Best Model:** {best_model.replace('_pipeline.joblib', '')} "
                  f"(RMSE: {results_summary[best_model]['RMSE']})")
    
    # # Add recommendations
    # st.markdown("**Recommendations:**")
    # if task_type == "classification":
    #     st.markdown("""
    #     - Consider class imbalance if precision/recall values vary significantly across classes
    #     - Evaluate if false positives or false negatives are more costly for your business case
    #     - For multi-class problems, examine per-class metrics to identify weak spots
    #     """)
    # else:
    #     st.markdown("""
    #     - Examine residual patterns to identify potential model improvements
    #     - Consider whether absolute or relative error is more important for your use case
    #     - Check for heteroscedasticity (varying error magnitude across prediction range)
    #     """)

#########################################
# Page 7: Export
#########################################

elif page == "Export":
    st.header("Export")
    ensure_session_state()
    inject_css()

    st.markdown("# Export Models & Project")
    st.caption("Download trained models, preprocessing pipelines, datasets, and complete project bundles")

    # Initialize session state for export settings
    if 'export_settings' not in st.session_state:
        st.session_state.export_settings = {
            'include_data': True,
            'include_splits': True,
            'include_reports': True,
            'include_requirements': True,
        }

    # Get current date for timestamping
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # --- Discover available models from session state ---
    def get_available_models():
        """Get models from session state"""
        models = []
        
        # Get individual models
        if 'trained_models' in st.session_state:
            for name, pipeline in st.session_state.trained_models.items():
                metrics = st.session_state.model_results.get(name, {})
                models.append({
                    'name': name,
                    'pipeline': pipeline,
                    'metrics': metrics,
                    'is_best': False
                })
        
        # Get best model
        if 'best_model' in st.session_state:
            best = st.session_state.best_model
            models.append({
                'name': f"BEST_{best['name']}",
                'pipeline': best['pipeline'],
                'metrics': best['metrics'],
                'is_best': True
            })
            
        return models

    # Get available models
    available_models = get_available_models()
    
    # Separate best model from others
    best_models = [model for model in available_models if model['is_best']]
    other_models = [model for model in available_models if not model['is_best']]
    
    # --- Export Options ---
    st.markdown("## Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Content Selection")
        include_data = st.checkbox("Include dataset", value=st.session_state.export_settings['include_data'], key ="7")
        include_splits = st.checkbox("Include train/test splits", value=st.session_state.export_settings['include_splits'], key ="8")
        include_reports = st.checkbox("Include reports", value=st.session_state.export_settings['include_reports'],key ="9")
        include_requirements = st.checkbox("Include requirements.txt", value=st.session_state.export_settings['include_requirements'], key ="10")
        
    with col2:
        st.markdown("### Model Selection")
        
        # Project metadata
        project_name = st.text_input("Project name", value="ml_project")
        version = st.text_input("Version", value=f"1.0_{current_date}")
        
        # Model selection
        if other_models:
            selected_model_names = st.multiselect(
                "Select models to export",
                options=[model['name'] for model in other_models],
                default=[model['name'] for model in other_models]
            )
            
            # Map back to actual model objects
            selected_models = []
            for name in selected_model_names:
                for model in other_models:
                    if model['name'] == name:
                        selected_models.append(model)
                        break
        else:
            st.info("No trained models found. Train models on the Pipeline page first.")
            selected_models = []

    # Update session state
    st.session_state.export_settings.update({
        'include_data': include_data,
        'include_splits': include_splits,
        'include_reports': include_reports,
        'include_requirements': include_requirements
    })

    # --- Individual Model Downloads ---
    st.markdown("## Individual Model Downloads")
    
    if selected_models:
        cols = st.columns(min(3, len(selected_models)))
        
        for i, model_info in enumerate(selected_models):
            with cols[i % 3]:
                try:
                    # Create model data for download
                    model_data = {
                        "pipeline": model_info['pipeline'],
                        "metrics": model_info['metrics'],
                        "model_name": model_info['name'],
                        "training_date": datetime.now().isoformat()
                    }
                    
                    # Convert to bytes
                    model_bytes = io.BytesIO()
                    joblib.dump(model_data, model_bytes)
                    model_bytes.seek(0)
                    
                    # Create descriptive filename
                    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_info["name"])
                    filename = f"{safe_name}_{current_date}.joblib"
                    
                    # Create download button
                    st.download_button(
                        label=f"Download {model_info['name']}",
                        data=model_bytes.getvalue(),
                        file_name=filename,
                        mime="application/octet-stream",
                        key=f"dl_{model_info['name']}"
                    )
                    
                    # Show model info
                    st.caption(f"Type: {type(model_info['pipeline']).__name__}")
                    if model_info.get('metrics'):
                        best_metric = next(iter(model_info['metrics'].values()), 'N/A')
                        st.caption(f"Best: {best_metric:.4f}" if isinstance(best_metric, (int, float)) else f"Metric: {best_metric}")
                
                except Exception as e:
                    st.error(f"Error preparing {model_info['name']}: {e}")
    else:
        st.info("No models selected for individual download")

    # --- Best Model Download ---
    st.markdown("## Best Model Download")
    
    if best_models:
        for best_model_info in best_models:
            try:
                # Create model data for download
                model_data = {
                    "pipeline": best_model_info['pipeline'],
                    "metrics": best_model_info['metrics'],
                    "model_name": best_model_info['name'],
                    "training_date": datetime.now().isoformat(),
                    "is_best_model": True
                }
                
                # Convert to bytes
                model_bytes = io.BytesIO()
                joblib.dump(model_data, model_bytes)
                model_bytes.seek(0)
                
                # Create filename
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', best_model_info["name"])
                filename = f"{safe_name}_{current_date}.joblib"
                
                st.download_button(
                    label=f"Download Best Model ({best_model_info['name']})",
                    data=model_bytes.getvalue(),
                    file_name=filename,
                    mime="application/octet-stream",
                    key=f"dl_best_{best_model_info['name']}"
                )
                
                # Show metrics if available
                if best_model_info.get('metrics'):
                    metrics_df = pd.DataFrame.from_dict(best_model_info['metrics'], orient='index', columns=['Value'])
                    st.dataframe(metrics_df.style.format("{:.4f}"))
                
                # Add some spacing between multiple best models
                st.markdown("---")
                    
            except Exception as e:
                st.error(f"Error preparing best model {best_model_info['name']}: {e}")
    else:
        st.info("No best model found. Train models first to generate a best model.")

    # --- All Models Bundle ---
    st.markdown("## All Models Bundle")
    
    if selected_models or best_models:
        bundle_filename = f"{project_name}_models_{current_date}.zip"
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add selected models
            for model_info in selected_models:
                try:
                    # Create model data
                    model_data = {
                        "pipeline": model_info['pipeline'],
                        "metrics": model_info['metrics'],
                        "model_name": model_info['name'],
                        "training_date": datetime.now().isoformat()
                    }
                    
                    # Add to zip
                    model_bytes = io.BytesIO()
                    joblib.dump(model_data, model_bytes)
                    model_bytes.seek(0)
                    
                    zip_file.writestr(
                        f"{model_info['name']}_{current_date}.joblib", 
                        model_bytes.getvalue()
                    )
                    
                    # Add metadata
                    metadata = {
                        'model_name': model_info['name'],
                        'metrics': model_info.get('metrics', {}),
                        'export_date': datetime.now().isoformat()
                    }
                    
                    zip_file.writestr(
                        f"{model_info['name']}_metadata.json", 
                        json.dumps(metadata, indent=2)
                    )
                        
                except Exception as e:
                    st.warning(f"Could not add {model_info['name']} to bundle: {e}")
            
            # Add best models
            for best_model_info in best_models:
                try:
                    # Create model data
                    model_data = {
                        "pipeline": best_model_info['pipeline'],
                        "metrics": best_model_info['metrics'],
                        "model_name": best_model_info['name'],
                        "training_date": datetime.now().isoformat(),
                        "is_best_model": True
                    }
                    
                    # Add to zip
                    model_bytes = io.BytesIO()
                    joblib.dump(model_data, model_bytes)
                    model_bytes.seek(0)
                    
                    zip_file.writestr(
                        f"{best_model_info['name']}_{current_date}.joblib", 
                        model_bytes.getvalue()
                    )
                    
                    # Add metadata
                    metadata = {
                        'model_name': best_model_info['name'],
                        'metrics': best_model_info.get('metrics', {}),
                        'is_best_model': True,
                        'export_date': datetime.now().isoformat()
                    }
                    
                    zip_file.writestr(
                        f"{best_model_info['name']}_metadata.json", 
                        json.dumps(metadata, indent=2)
                    )
                        
                except Exception as e:
                    st.warning(f"Could not add best model {best_model_info['name']} to bundle: {e}")
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="Download All Models as ZIP",
            data=zip_buffer.getvalue(),
            file_name=bundle_filename,
            mime="application/zip",
            key="dl_all_models"
        )
    else:
        st.info("No models available for bundle")

    # --- Complete Project Export ---
    st.markdown("## Complete Project Export")
    
    # Collect all files to include
    all_files_to_include = []
    
    # Add models
    for model_info in selected_models + best_models:
        # Create model data
        model_data = {
            "pipeline": model_info['pipeline'],
            "metrics": model_info['metrics'],
            "model_name": model_info['name'],
            "training_date": datetime.now().isoformat(),
            "is_best_model": model_info.get('is_best', False)
        }
        
        # Convert to bytes
        model_bytes = io.BytesIO()
        joblib.dump(model_data, model_bytes)
        model_bytes.seek(0)
        
        all_files_to_include.append((f"models/{model_info['name']}_{current_date}.joblib", model_bytes.getvalue()))
    
    # Add data if requested and available
    if include_data and st.session_state.df is not None:
        # Save current dataset
        dataset_filename = f"{project_name}_dataset_{current_date}.csv"
        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
        all_files_to_include.append((f"data/{dataset_filename}", csv_data))
    
    # Add splits if requested and available
    if include_splits and "split_result" in st.session_state:
        splits = st.session_state["split_result"]
        for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
            if key in splits and splits[key] is not None:
                if isinstance(splits[key], pd.DataFrame):
                    split_data = splits[key].to_csv(index=False).encode('utf-8')
                else:
                    split_data = splits[key].to_frame().to_csv(index=False).encode('utf-8')
                all_files_to_include.append((f"splits/{key}.csv", split_data))
    
    # Create project metadata
    project_metadata = {
        'project_name': project_name,
        'version': version,
        'export_date': datetime.now().isoformat(),
        'dataset_shape': list(st.session_state.df.shape) if st.session_state.df is not None else None,
        'included_models': [model_info['name'] for model_info in selected_models + best_models],
        'export_settings': st.session_state.export_settings
    }
    
    if st.button("Create Complete Project Export",key= "49"):
        if not all_files_to_include:
            st.warning("No files selected for export")
        else:
            with st.spinner("Creating project bundle..."):
                # Create ZIP in memory
                project_zip_buffer = io.BytesIO()
                with zipfile.ZipFile(project_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add all files
                    for filename, data in all_files_to_include:
                        zip_file.writestr(filename, data)
                    
                    # Add requirements if requested
                    if include_requirements:
                        try:
                            import pkg_resources
                            requirements = "\n".join(
                                sorted([f"{pkg.project_name}=={pkg.version}" for pkg in pkg_resources.working_set])
                            )
                            zip_file.writestr("requirements.txt", requirements)
                        except Exception as e:
                            st.warning(f"Could not generate requirements.txt: {e}")
                    
                    # Add metadata
                    zip_file.writestr(
                        "project_metadata.json", 
                        json.dumps(project_metadata, indent=2)
                    )
                
                project_zip_buffer.seek(0)
                zip_data = project_zip_buffer.getvalue()
                
                # Create download button
                project_filename = f"{project_name}_complete_{current_date}.zip"
                
                st.download_button(
                    label="Download Complete Project",
                    data=zip_data,
                    file_name=project_filename,
                    mime="application/zip",
                    key="dl_complete_project"
                )
                
                st.success("Project bundle created successfully!")
                
                # Show summary
                st.markdown("### Export Summary")
                summary_data = {
                    'Models': len(selected_models + best_models),
                    'Dataset': 'Included' if include_data else 'Excluded',
                    'Splits': 'Included' if include_splits else 'Excluded',
                    'Requirements': 'Included' if include_requirements else 'Excluded',
                    'Total Files': len(all_files_to_include) + (1 if include_requirements else 0) + 1  # +1 for metadata
                }
                st.table(pd.DataFrame.from_dict(summary_data, orient='index', columns=['Count']))

    st.markdown("---")
    st.warning("**Note:** Always download your exported files before closing the application. Files are not preserved between sessions.")


###########################################
# Page 8: Prediction
###########################################
elif page == "Prediction":
    st.header("Prediction")
    ensure_session_state()
    inject_css()

    st.markdown("# Prediction")
    st.caption("Load saved models and pipelines, perform batch or single-record inference, monitor drift.")

    # --- Model Loader ---
    # Get models from both session state and disk
    model_files = glob.glob("*.joblib") + glob.glob("*.pkl")
    model_files = [f for f in model_files if not f.startswith('.')]
    
    # Get current date for filtering
    current_date = datetime.now().strftime('%Y%m%d')
    
    # Filter models from current session (created today)
    current_session_models = []
    for model_file in model_files:
        try:
            creation_time = datetime.fromtimestamp(os.path.getctime(model_file))
            if creation_time.strftime('%Y%m%d') == current_date:
                current_session_models.append(model_file)
        except OSError:
            # If we can't get creation time, include the file
            current_session_models.append(model_file)
    
    # If no current session models, show all models
    if not current_session_models:
        current_session_models = model_files
    
    # Get models from session state
    session_models = {}
    if 'trained_models' in st.session_state:
        for name, pipeline in st.session_state.trained_models.items():
            metrics = st.session_state.model_results.get(name, {})
            session_models[name] = {
                'pipeline': pipeline,
                'metrics': metrics,
                'is_best': False,
                'source': 'session'
            }
    
    # Get best model from session state
    if 'best_model' in st.session_state:
        best = st.session_state.best_model
        session_models[f"BEST_{best['name']}"] = {
            'pipeline': best['pipeline'],
            'metrics': best['metrics'],
            'is_best': True,
            'source': 'session'
        }
    
    # Create model selection options
    session_model_names = list(session_models.keys())
    file_model_names = current_session_models
    
    st.markdown("## Model Selection")
    
    if not session_model_names and not file_model_names:
        st.warning("No trained models found. Please train models on the Pipeline page first.")
    
    # --- Model Comparison Table ---
    if session_model_names:
        st.markdown("### Model Performance Comparison")
        
        # Create comparison table
        comparison_data = []
        for name, model_info in session_models.items():
            row = {'Model': name}
            row.update(model_info['metrics'])
            row['Best'] = '✓' if model_info['is_best'] else ''
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        if not comparison_df.empty:
            # Style the comparison table
            styled_df = comparison_df.style.highlight_max(
                subset=[col for col in comparison_df.columns if col not in ['Model', 'Best']], 
                color='lightgreen'
            ).highlight_min(
                subset=[col for col in comparison_df.columns if col not in ['Model', 'Best']], 
                color='lightcoral'
            )
            
            st.dataframe(styled_df, use_container_width=True)
    
    # --- Model Selection Interface ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create dropdown options
        dropdown_options = []
        
        # Add session models first
        if session_model_names:
            dropdown_options.append("--- Session Models ---")
            dropdown_options.extend(session_model_names)
        
        # Add file models
        if file_model_names:
            dropdown_options.append("--- Saved Model Files ---")
            dropdown_options.extend(file_model_names)
        
        selected_model = st.selectbox(
            "Choose a model for prediction",
            options=dropdown_options,
            help="Select from models in current session or saved model files"
        )
    
    with col2:
        st.markdown("### Or upload new model")
        uploaded_model_file = st.file_uploader(
            "Upload model file", 
            type=["joblib", "pkl"],
            help="Upload a .joblib or .pkl file from your computer",
            key="prediction_upload"
        )

    # Show option to view all models (not just current session)
    if st.checkbox("Show all available model files", key ="16"):
        if model_files:
            st.info("All available model files:")
            for model in model_files:
                try:
                    creation_time = datetime.fromtimestamp(os.path.getctime(model))
                    st.write(f"• {model} (created: {creation_time.strftime('%Y-%m-%d %H:%M')})")
                except OSError:
                    st.write(f"• {model} (creation time unavailable)")
        else:
            st.info("No model files found on disk.")

    # --- Load selected model ---
    loaded_model, meta_metrics, target_classes, model_name, task_type = None, None, None, None, None
    
    if selected_model and selected_model not in ["--- Session Models ---", "--- Saved Model Files ---"]:
        # Check if it's a session model
        if selected_model in session_models:
            model_info = session_models[selected_model]
            loaded_model = model_info['pipeline']
            meta_metrics = model_info['metrics']
            model_name = selected_model
            st.success(f"Loaded session model: {selected_model}")
            
            # Try to determine task type
            if loaded_model and hasattr(loaded_model, 'predict'):
                # Better task type detection
                try:
                    # Check if we have target information from session state
                    if hasattr(st.session_state, 'target_column') and st.session_state.target_column in st.session_state.df.columns:
                        y_sample = st.session_state.df[st.session_state.target_column]
                        
                        # More robust task detection
                        if pd.api.types.is_numeric_dtype(y_sample):
                            # For regression: many unique values, mostly numeric
                            unique_ratio = y_sample.nunique() / len(y_sample)
                            if unique_ratio > 0.5 or y_sample.nunique() > 20:
                                task_type = "regression"
                            else:
                                # Could be classification with numeric labels
                                task_type = "classification"
                        else:
                            # Non-numeric target is definitely classification
                            task_type = "classification"
                    else:
                        # Fallback: use model's predict method signature or known model types
                        model_type = str(type(loaded_model)).lower()
                        if any(reg_keyword in model_type for reg_keyword in ['regressor', 'regression', 'linear', 'lasso', 'ridge']):
                            task_type = "regression"
                        elif any(clf_keyword in model_type for clf_keyword in ['classifier', 'classification', 'logistic']):
                            task_type = "classification"
                        else:
                            # Final fallback: check if model has predict_proba (classification)
                            if hasattr(loaded_model, "predict_proba"):
                                task_type = "classification"
                            else:
                                task_type = "regression"  # Assume regression as default
                except Exception as e:
                    st.warning(f"Could not automatically determine task type: {e}")
                    task_type = "unknown"
                
        else:
            # It's a file model
            try:
                saved_obj = joblib.load(selected_model)
                st.success(f"Loaded model from: {selected_model}")
                
                # Show model creation time
                try:
                    creation_time = datetime.fromtimestamp(os.path.getctime(selected_model))
                    st.caption(f"Model created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
                except OSError:
                    st.caption("Creation time unavailable")
                
                # Extract model information
                if isinstance(saved_obj, dict) and "pipeline" in saved_obj:
                    loaded_model = saved_obj["pipeline"]
                    meta_metrics = saved_obj.get("metrics")
                    target_classes = saved_obj.get("classes")
                    model_name = saved_obj.get("model_name", "Unknown Model")
                    task_type = saved_obj.get("task_type")
                else:
                    loaded_model = saved_obj
                    model_name = "Direct Pipeline"
                
            except Exception as e:
                st.error(f"Failed to load {selected_model}: {e}")
    
    elif uploaded_model_file:
        try:
            with open("temp_uploaded_model.joblib", "wb") as f:
                f.write(uploaded_model_file.getbuffer())
            saved_obj = joblib.load("temp_uploaded_model.joblib")
            st.success(f"Uploaded model loaded successfully: {uploaded_model_file.name}")
            
            # Extract model information
            if isinstance(saved_obj, dict) and "pipeline" in saved_obj:
                loaded_model = saved_obj["pipeline"]
                meta_metrics = saved_obj.get("metrics")
                target_classes = saved_obj.get("classes")
                model_name = saved_obj.get("model_name", "Unknown Model")
                task_type = saved_obj.get("task_type")
            else:
                loaded_model = saved_obj
                model_name = "Direct Pipeline"
                
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")
    
    # Display model information
    if loaded_model:
        st.markdown("### Model Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Name:** {model_name}")
            st.info(f"**Type:** {type(loaded_model).__name__}")
            
        with col2:
            if task_type:
                st.info(f"**Task:** {task_type}")
            if meta_metrics:
                best_metric = next(iter(meta_metrics.values()), 'N/A')
                metric_text = f"{best_metric:.4f}" if isinstance(best_metric, (int, float)) else f"{best_metric}"
                st.info(f"**Best Metric:** {metric_text}")
        
        # Show detailed metrics if available
        if meta_metrics:
            with st.expander("View Detailed Metrics"):
                metrics_df = pd.DataFrame.from_dict(meta_metrics, orient='index', columns=['Value'])
                st.dataframe(metrics_df.style.format("{:.4f}"))

    # --- Input Interface ---
    if loaded_model:
        st.markdown("## Input Data")
        mode = st.radio("Prediction Mode", ["Single Record", "Batch Upload"])
        
        if mode == "Single Record":
            if st.session_state.df is not None:
                # Get feature columns from training data
                feature_columns = st.session_state.df.columns.tolist()
                
                # Remove target column if it exists in session state
                if hasattr(st.session_state, 'target_column') and st.session_state.target_column in feature_columns:
                    feature_columns.remove(st.session_state.target_column)
                
                st.markdown("### Fill values for prediction:")
                input_values = {}
                
                for col in feature_columns:
                    col_data = st.session_state.df[col]
                    
                    # Determine input type based on column data
                    if col_data.dtype == 'object' or col_data.nunique() < 20:
                        # Categorical column - use dropdown
                        unique_vals = col_data.dropna().unique().tolist()
                        if len(unique_vals) > 0:
                            default_val = unique_vals[0]
                            input_val = st.selectbox(
                                f"{col}",
                                options=unique_vals,
                                index=0,
                                help=f"Select from available {col} values",
                                key=f"input_{col}"
                            )
                        else:
                            input_val = st.text_input(f"{col}", value="", key=f"input_{col}")
                    elif pd.api.types.is_numeric_dtype(col_data):
                        # Numeric column - use number input with free input
                        # Show example value from training data as placeholder
                        example_val = float(col_data.mean()) if not col_data.empty and not pd.isna(col_data.mean()) else 0.0
                        input_val = st.number_input(
                            f"{col}",
                            value=example_val,
                            step=0.1,
                            help=f"Enter any numeric value for {col}",
                            key=f"input_{col}"
                        )
                    elif pd.api.types.is_datetime64_any_dtype(col_data):
                        # Date column - use date input
                        min_date = col_data.min() if not col_data.empty else datetime.now().date()
                        max_date = col_data.max() if not col_data.empty else datetime.now().date()
                        input_val = st.date_input(f"{col}", value=min_date, min_value=min_date, max_value=max_date, key=f"input_{col}")
                        input_val = input_val.isoformat()
                    else:
                        # Fallback to text input
                        input_val = st.text_input(f"{col}", value="", key=f"input_{col}")
                    
                    input_values[col] = input_val

                if st.button("Predict Single",key= "50"):
                    try:
                        # Create DataFrame with proper data types
                        row_df = pd.DataFrame([input_values])
                        
                        # Convert to appropriate data types based on training data
                        for col in row_df.columns:
                            if col in st.session_state.df.columns:
                                if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                                    row_df[col] = pd.to_numeric(row_df[col], errors='coerce')
                                elif pd.api.types.is_datetime64_any_dtype(st.session_state.df[col]):
                                    row_df[col] = pd.to_datetime(row_df[col], errors='coerce')
                        
                        # Make prediction
                        pred = loaded_model.predict(row_df)
                        
                        # Decode prediction to class name if available
                        if target_classes is not None and hasattr(pred, '__iter__'):
                            try:
                                pred_value = pred[0]
                                if isinstance(pred_value, (int, float, np.number)):
                                    pred_index = int(pred_value)
                                    if 0 <= pred_index < len(target_classes):
                                        prediction_display = target_classes[pred_index]
                                    else:
                                        prediction_display = str(pred_value)
                                else:
                                    prediction_display = str(pred_value)
                            except (ValueError, IndexError, TypeError):
                                prediction_display = str(pred[0])
                        else:
                            prediction_display = str(pred[0])
                        
                        # Display prediction with nice formatting
                        st.success("### Prediction Result")
                        st.markdown(f"**Predicted Value:** `{prediction_display}`")
                        
                        # Show probabilities if available
                        if hasattr(loaded_model, "predict_proba"):
                            proba = loaded_model.predict_proba(row_df)
                            if target_classes is not None:
                                proba_df = pd.DataFrame(proba, columns=target_classes)
                            else:
                                proba_df = pd.DataFrame(proba, columns=[f"Class {i}" for i in range(proba.shape[1])])
                            
                            st.markdown("### Prediction Probabilities")
                            # Format probabilities as percentages
                            proba_df_display = proba_df.copy()
                            for col in proba_df_display.columns:
                                proba_df_display[col] = proba_df_display[col].apply(lambda x: f"{x:.2%}")
                            
                            st.dataframe(proba_df_display.style.highlight_max(axis=1, color="lightgreen"))
                            
                            # Show probability distribution chart
                            if len(proba_df.columns) > 1:
                                fig = px.bar(
                                    proba_df.T.reset_index(), 
                                    x='index', 
                                    y=0,
                                    title="Prediction Probability Distribution",
                                    labels={'index': 'Class', '0': 'Probability'}
                                )
                                fig.update_layout(yaxis_tickformat='.0%')
                                st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.error(f"Error details: {str(e)}")
            else:
                st.info("No dataset available to infer feature columns. Upload dataset on Data Loading page.")
        
        else:  # Batch mode
            batch_file = st.file_uploader("Upload batch file (CSV/Excel)", type=["csv","xlsx"], key="batch_upload")
            if batch_file is not None:
                try:
                    if batch_file.name.endswith(".csv"):
                        input_df = pd.read_csv(batch_file)
                    else:
                        input_df = pd.read_excel(batch_file)
                    st.write("Batch file preview:", input_df.head())
                except Exception as e:
                    st.error(f"Failed to load batch file: {e}")

                if st.button("Predict Batch",key= "52"):
                    try:
                        with st.spinner("Making predictions..."):
                            preds = loaded_model.predict(input_df)
                            
                            # Decode predictions to class names if available
                            if target_classes is not None:
                                decoded_preds = []
                                for pred in preds:
                                    try:
                                        if isinstance(pred, (int, float, np.number)):
                                            pred_index = int(pred)
                                            if 0 <= pred_index < len(target_classes):
                                                decoded_preds.append(target_classes[pred_index])
                                            else:
                                                decoded_preds.append(str(pred))
                                        else:
                                            decoded_preds.append(str(pred))
                                    except (ValueError, IndexError, TypeError):
                                        decoded_preds.append(str(pred))
                                preds = decoded_preds
                            
                            out_df = input_df.copy()
                            out_df["Prediction"] = preds

                            # Add probabilities if available
                            if hasattr(loaded_model, "predict_proba"):
                                proba = loaded_model.predict_proba(input_df)
                                if target_classes is not None:
                                    proba_df = pd.DataFrame(proba, columns=[f"Prob_{cls}" for cls in target_classes])
                                else:
                                    proba_df = pd.DataFrame(proba, columns=[f"Prob_Class_{i}" for i in range(proba.shape[1])])
                                
                                # Format probabilities as percentages
                                for col in proba_df.columns:
                                    proba_df[col] = proba_df[col].apply(lambda x: f"{x:.2%}")
                                
                                out_df = pd.concat([out_df, proba_df], axis=1)

                            st.success(f"Predictions completed for {len(out_df)} records")
                            st.dataframe(out_df.head())

                            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📥 Download Predictions CSV", 
                                data=csv_bytes, 
                                file_name="predictions.csv", 
                                mime="text/csv",
                                key="download_predictions"
                            )
                            
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")
                        st.error(f"Error details: {str(e)}")
    
    else:
        st.warning("Please load a trained model first.")

    st.markdown("---")
    st.warning("**Note:** Make sure your input data has the same features and data types as the training data.")
