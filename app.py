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

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import xgboost as xgb
import lightgbm as lgb
import optuna
import plotly.express as px
import plotly.graph_objects as go

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

load_dotenv()

# Initialize Groq client
def init_groq_client():
    """Initialize Groq client with API key from environment or user input"""
    if not GROQ_AVAILABLE:
        return None
    
    # Try to get from Streamlit secrets first
    try:
        api_key = st.secrets.get('GROQ_API_KEY', '')
    except:
        api_key = ''
    
    # Fallback to hardcoded key if not in secrets
    if not api_key:
        api_key = "sk_a8vNuNJxbNCvZOl5MAQ6WGdyb3FYPIdxOADqLz5A5G2TGLhdk00u"
    
    if 'groq_client' not in st.session_state:
        if api_key:
            try:
                st.session_state.groq_client = Groq(api_key=api_key)
                st.session_state.groq_available = True
            except Exception as e:
                st.error(f"Failed to initialize Groq client: {e}")
                st.session_state.groq_available = False
        else:
            st.session_state.groq_available = False
    
    return st.session_state.get('groq_client')
# LLM helper functions
def call_groq_llm(prompt, model="llama3-70b-8192", max_tokens=1024, temperature=0.7):
    """Call Groq LLM API with the given prompt"""
    client = init_groq_client()
    if not client:
        return "Groq API not available. Please set GROQ_API_KEY environment variable."
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stream=False,
            stop=None,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
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
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Data Loading",
        "EDA",
        "Train-Test Split",
        "Pipeline",
        "Training",
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
    st.title("ML Playground - Step by Step Guide")
    st.write("Welcome to the Machine Learning Playground! Follow these steps to build, train, and download your ML models.")
    
    ensure_session_state()
    inject_css()
    
    # Initialize Groq
    init_groq_client()
    if st.session_state.groq_available:
        st.sidebar.success("🤖 Groq LLM Available")
    else:
        st.sidebar.warning("⚠️ Groq LLM Not Configured")

    # Step-by-step guide
    st.markdown("## Step-by-Step Workflow")
    
    # Step 1: Data Loading
    with st.expander("Step 1: Load Your Data", expanded=True):
        st.markdown("""
        **Start by loading your dataset:**
        
        - **Upload File**: CSV or Excel files from your computer
        - **Preloaded Dataset**: Sample datasets (Iris, Titanic, Diabetes, etc.)
        - **From URL**: Load data from a public URL
        - **SQL Database**: Connect to your database
        - **Google Sheets**: Import from Google Sheets
        - **Kaggle**: Download datasets from Kaggle
        
        **→ Go to:** **Data Loading** page in the sidebar
        """)
        
        # Quick upload option on home page
        st.markdown("---")
        st.markdown("### Quick Start: Upload Your Data")
        new_df = upload_panel(key_prefix="home")
        if new_df is not None:
            st.session_state.df = new_df
            st.session_state.dataset_name = getattr(new_df, "_dataset_name", None) or "uploaded_or_sample"
            st.session_state.versions.append(("loaded", new_df.copy()))
            st.success("✅ Dataset loaded! You can now proceed to Step 2.")

    # Step 2: EDA
    with st.expander("Step 2: Explore & Clean Your Data"):
        st.markdown("""
        **Explore and prepare your data:**
        
        - **Data Overview**: Summary statistics, missing values, data types
        - **Data Cleaning**: Handle missing values, outliers, duplicates
        - **Feature Engineering**: Create new features, transform variables
        - **Visualizations**: Interactive charts and plots
        - **Text Processing**: Clean and preprocess text columns
        - **Automated Profiling**: Comprehensive data profile report
        
        **→ Go to:** **EDA** page in the sidebar
        
        **💡 Tip**: Use the snapshot feature to compare before/after changes
        """)

    # Step 3: Train-Test Split
    with st.expander("Step 3: Split Your Data"):
        st.markdown("""
        **Prepare for model training:**
        
        - **Select Target Column**: Choose what you want to predict
        - **Split Parameters**: Set test size, random state, validation split
        - **Download Splits**: Export train/validation/test sets
        
        **→ Go to:** **Train-Test Split** page in the sidebar
        
        **💡 Tip**: Typical splits are 70-80% training, 20-30% testing
        """)

    # Step 4: Pipeline Building
    with st.expander("Step 4: Build Your Pipeline"):
        st.markdown("""
        **Create preprocessing and modeling pipeline:**
        
        - **Column Assignment**: Specify numeric vs categorical features
        - **Numeric Pipeline**: Imputation, scaling strategies
        - **Categorical Pipeline**: Encoding, imputation methods
        - **Model Selection**: Choose algorithms for your problem type
        - **Train Models**: Fit multiple models and compare performance
        
        **→ Go to:** **Pipeline** page in the sidebar
        
        **💡 Tip**: Start with simple models first, then try more complex ones
        """)

    # Step 5: Training & Tuning
    with st.expander("Step 5: Train & Optimize Models"):
        st.markdown("""
        **Fine-tune your models:**
        
        - **Hyperparameter Tuning**: Grid search, random search, Optuna
        - **Cross-Validation**: K-fold validation for robust evaluation
        - **Early Stopping**: Prevent overfitting
        - **Performance Metrics**: Track model performance
        
        **→ Go to:** **Training** page in the sidebar
        
        **Tip**: Use different search strategies for optimal results
        """)

    # Step 6: Evaluation
    with st.expander("Step 6: Evaluate Model Performance"):
        st.markdown("""
        **Comprehensive model evaluation:**
        
        - **Performance Metrics**: Accuracy, Precision, Recall, F1, R², etc.
        - **Confusion Matrix**: Visualize classification performance
        - **ROC Curves**: Analyze model discrimination ability
        - **SHAP Analysis**: Understand feature importance
        - **Model Comparison**: Compare multiple models side-by-side
        
        **→ Go to:** **Final Evaluation** page in the sidebar
        
        **Tip**: Look at multiple metrics to get a complete picture
        """)

    # Step 7: Export & Deployment
    with st.expander("Step 7: Export & Deploy"):
        st.markdown("""
        **Package and deploy your solution:**
        
        - **Model Export**: Save trained models and pipelines
        - **Bundle Creation**: Package everything into a ZIP file
        - **Metadata**: Add project information and documentation
        - **Requirements**: Generate dependency list
        
        **→ Go to:** **Export** page in the sidebar
        
        **💡 Tip**: Always include a requirements.txt for reproducibility
        """)

    # Step 8: Prediction
    with st.expander("Step 8: Make Predictions"):
        st.markdown("""
        **Use your trained models for inference:**
        
        - **Load Models**: Select from previously trained models
        - **Single Prediction**: Input values for one prediction
        - **Batch Prediction**: Upload file for multiple predictions
        - **Class Decoding**: Get human-readable predictions
        - **Probability Scores**: See prediction confidence levels
        
        **→ Go to:** **Prediction** page in the sidebar
        
        **Tip**: Test your model with edge cases to ensure robustness
        """)

    # Quick status dashboard
    st.markdown("---")
    st.markdown("## Current Project Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.df is not None:
            st.success("Data Loaded")
            st.caption(f"{st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
        else:
            st.warning("No Data")
            st.caption("Complete Step 1")
    
    with col2:
        if "split_result" in st.session_state:
            st.success("Data Split")
            split = st.session_state["split_result"]
            train_size = len(split["X_train"])
            test_size = len(split["X_test"])
            st.caption(f"Train: {train_size}, Test: {test_size}")
        else:
            st.info("Ready for Split")
            st.caption("Complete Step 3")
    
    with col3:
        model_files = glob.glob("*.joblib") + glob.glob("*.pkl")
        if model_files:
            st.success("Models Trained")
            st.caption(f"{len(model_files)} model(s) available")
        else:
            st.info("Ready for Training")
            st.caption("Complete Step 4-5")

    # Quick actions
    st.markdown("---")
    st.markdown("## Quick Actions")
    
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.session_state.df is not None:
            preview_card(st.session_state.df, st.session_state.get("dataset_name"))
    
    with quick_col2:
        if st.session_state.df is not None:
            download_panel(st.session_state.df, filename_basename=st.session_state.get("dataset_name") or "dataset")
    
    with quick_col3:
        st.markdown("### Need Help?")
        st.markdown("""
        - Check each step's instructions
        - Use the sidebar to navigate
        - Save your work frequently
        - Hover over options for tooltips
        """)
        
        # AI Assistant
        if st.session_state.groq_available:
            with st.expander("🤖 AI Assistant"):
                user_question = st.text_input("Ask a question about ML workflow:")
                if user_question and st.button("Get AI Help"):
                    with st.spinner("Thinking..."):
                        response = call_groq_llm(f"""
                        You're an ML expert assistant. Answer this question about machine learning workflow:
                        {user_question}
                        
                        Provide a concise, helpful response focused on practical advice.
                        """)
                        st.info(response)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built for YOU using Streamlit | ML Playground v1.0</p>
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
        elif source == "Preloaded Dataset":
            preload_choice = st.selectbox(
                "Choose dataset",
                ["(choose)", "sns:tips", "sns:iris", "sns:titanic",
                 "sk:iris", "sk:wine", "sk:breast_cancer", "sk:diabetes", "sk:digits"]
            )
            if st.button("Load Preloaded Dataset") and preload_choice != "(choose)":
                df, name = load_sample_dataset(preload_choice)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.dataset_name = name
                    st.session_state.versions.append(("preloaded", df.copy()))
                    st.success(f"Loaded {name}! Shape: {df.shape}")

        # --- Option 3: From URL ---
        elif source == "From URL":
            url = st.text_input("Enter URL to CSV or Excel file")
            if st.button("Load from URL") and url:
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
            if st.button("Load from SQL") and db_uri and query:
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
            if st.button("Load from Google Sheets") and sheet_url:
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

            if kaggle_file and dataset_id and st.button("Download & Load from Kaggle"):
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
                if st.button("Analyze Dataset with AI"):
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
    st.warning("**Reminder:** Save your progress by downloading the dataset before closing. Do not rename the file if you plan to reload it later.")


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
        with st.expander("## AI-Powered Data Analysis", expanded=False):
            analysis_type = st.selectbox("Analysis Type", 
                                       ["General Overview", "Data Quality", "Feature Suggestions", 
                                        "ML Readiness", "Custom Question"])
            
            custom_question = ""
            if analysis_type == "Custom Question":
                custom_question = st.text_input("Ask a specific question about your data")
            
            if st.button("Run AI Analysis"):
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

        st.button("Snapshot current dataset for Before/After", on_click=snapshot_current, kwargs={"label": "overview"})

        

    # ---------- combine/append/merge ----------
    with st.expander("Combine Datasets (append / merge)", expanded=False):
        tabs = st.tabs(["Append (rows)", "Merge (SQL-style)"])
        with tabs[0]:
            files = st.file_uploader("Upload CSV/Excel to append (multiple allowed)", type=["csv","xlsx"], accept_multiple_files=True)
            if files and st.button("Append to current"):
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
                    
                    if st.button("Perform Merge"):
                        merged = pd.merge(df, merge_df, left_on=left_key, right_on=right_key, how=join_type)
                        st.session_state[DF_KEY] = merged
                        st.success(f"Merged dataset shape: {merged.shape}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Merge failed: {e}")

    # ---------- data cleaning ----------
    with st.expander("Data Cleaning", expanded=False):
        tabs = st.tabs(["Missing Values", "Outliers", "Duplicates", "Data Types", "Text Cleaning"])
        
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
                    if st.button("Drop rows with missing values"):
                        st.session_state[DF_KEY] = df.dropna(subset=[col])
                        st.success(f"Dropped {missing_count} rows")
                        
                elif strategy == "Fill with mean/median":
                    if df[col].dtype in ['int64', 'float64']:
                        fill_val = st.selectbox("Fill with", ["mean", "median"])
                        if st.button("Fill missing values"):
                            fill_value = df[col].mean() if fill_val == "mean" else df[col].median()
                            st.session_state[DF_KEY][col] = df[col].fillna(fill_value)
                            st.success(f"Filled with {fill_val}: {fill_value:.2f}")
                    else:
                        st.warning("Column must be numeric for mean/median imputation")
                        
                elif strategy == "Fill with mode":
                    if st.button("Fill with mode"):
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
                        st.session_state[DF_KEY][col] = df[col].fillna(mode_val)
                        st.success(f"Filled with mode: {mode_val}")
                        
                elif strategy == "Fill with custom value":
                    custom_val = st.text_input("Custom value to fill")
                    if st.button("Fill with custom value"):
                        st.session_state[DF_KEY][col] = df[col].fillna(custom_val)
                        st.success(f"Filled with: {custom_val}")
                        
                elif strategy == "Interpolate":
                    if df[col].dtype in ['int64', 'float64']:
                        if st.button("Interpolate missing values"):
                            st.session_state[DF_KEY][col] = df[col].interpolate()
                            st.success("Applied interpolation")
                    else:
                        st.warning("Interpolation only works for numeric columns")
                        
                elif strategy == "KNN Imputation":
                    if st.button("Apply KNN Imputation (nearest neighbors)"):
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
                    
                    if action == "Remove outliers" and st.button("Remove Outliers"):
                        if method == "IQR (Interquartile Range)":
                            st.session_state[DF_KEY] = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                        st.success(f"Removed {len(outliers)} outliers")
                        
                    elif action == "Cap outliers" and st.button("Cap Outliers"):
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
                if st.button("Remove Duplicate Rows"):
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
            
            if st.button("Convert Data Type") and new_type != "Keep as is":
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
                
                if st.button("Apply Text Cleaning"):
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
            if st.button("Create Column") and expr:
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
                n_bins = st.slider("Number of bins", 2, 20, 5)
                if st.button("Create Bins"):
                    st.session_state[DF_KEY][new_col_name] = pd.cut(df[bin_col], bins=n_bins, labels=False)
                    
            elif bin_method == "Equal frequency":
                n_bins = st.slider("Number of bins", 2, 20, 5)
                if st.button("Create Bins"):
                    st.session_state[DF_KEY][new_col_name] = pd.qcut(df[bin_col], q=n_bins, labels=False, duplicates='drop')
                    
            elif bin_method == "Custom bins":
                bin_edges = st.text_input("Bin edges (comma separated)", "0,25,50,75,100")
                if st.button("Create Bins"):
                    edges = [float(x.strip()) for x in bin_edges.split(',')]
                    st.session_state[DF_KEY][new_col_name] = pd.cut(df[bin_col], bins=edges, labels=False)
        
        elif operation == "One-hot encoding":
            cat_col = st.selectbox("Categorical column", df.select_dtypes(exclude=np.number).columns.tolist())
            if st.button("One-hot encode"):
                dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
                st.session_state[DF_KEY] = pd.concat([df, dummies], axis=1)
                st.success(f"Created {len(dummies.columns)} one-hot encoded columns")
        
        elif operation == "Date extraction":
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if date_cols:
                date_col = st.selectbox("Date column", date_cols)
                extract = st.multiselect("Extract components", 
                                       ["Year", "Month", "Day", "Dayofweek", "Quarter", "Is weekend"])
                if st.button("Extract Date Components"):
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
                if st.button("Create Text Length Feature"):
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
        
        if st.button("Generate Chart"):
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
            threshold = st.slider("Correlation threshold", 0.5, 1.0, 0.8, 0.05)
            
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
            if st.button("Generate Profile Report"):
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
        if st.button("Snapshot Current State"):
            snapshot_current("manual_snapshot")
    
    with col2:
        if st.button("Reset to Original"):
            if st.session_state.versions:
                original_df = st.session_state.versions[0][1]
                st.session_state[DF_KEY] = original_df.copy()
                st.success("Reset to original dataset")
    
    with col3:
        download_panel(df, filename_basename="cleaned_dataset")

    st.warning("**Note:** All changes are applied to the current session. Download your cleaned dataset to preserve changes.")

################################################
# Page 3: Train-Test-Split
################################################

# ========================= pages/3_TrainTestSplit.py =========================

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
    target_col = st.selectbox("Select target column", [None] + df.columns.tolist())
    if not target_col:
        st.info("Please select a target column to continue.")
        st.stop()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ---------------- Parameters ----------------
    st.markdown("### Split Parameters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        test_size = st.slider("Test size (%)", 5, 50, 20, step=5) / 100.0
    with col2:
        shuffle = st.checkbox("Shuffle", value=True)
    with col3:
        random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    with col4:
        use_val = st.checkbox("Also create Validation set", value=False)

    val_size = None
    if use_val:
        val_size = st.slider("Validation size (%) of train", 5, 50, 20, step=5) / 100.0

    # ---------------- Split Action ----------------
    if st.button("Perform Split"):
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
        )

        if use_val:
            # Split train into train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=val_size, shuffle=shuffle, random_state=random_state
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
        }
        st.success(f"Split performed (saved as version {ts})")

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
            "Features": [split["X_train"].shape[1]]*3,
        }
        st.dataframe(pd.DataFrame(summary))

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

######################################
# Page 4: Pipeline & Modeling
#######################################

elif page == "Pipeline":
    st.header("Pipeline")
    ensure_session_state()
    inject_css()

    st.markdown("# ColumnTransformer + Pipeline Builder")
    st.caption("Interactively create preprocessing pipelines for numeric & categorical features, then train models.")

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

    # --- Column assignment
    st.markdown("## Assign Columns")
    num_cols = st.multiselect("Numeric Columns", cols, default=X_train.select_dtypes(include=np.number).columns.tolist())
    cat_cols = st.multiselect("Categorical Columns", [c for c in cols if c not in num_cols], default=X_train.select_dtypes(exclude=np.number).columns.tolist())

    st.info(f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

    # --- Numeric Pipeline Builder
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

    # --- Categorical Pipeline Builder
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

    # --- Build final preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="passthrough"
    )

    st.success("Preprocessor ready. Now choose models below.")

    # --- Model Training
    st.markdown("## Model Training")
    task_type = "classification" if y_train.nunique() < 20 and y_train.dtype != "float" else "regression"
    st.info(f"Detected Task Type: **{task_type}**")

    models = {}
    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "SVM": SVC(probability=True),
            "XGBoost": xgb.XGBClassifier(eval_metric="logloss"),
            "LightGBM": lgb.LGBMClassifier()
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Random Forest": RandomForestRegressor(),
            "SVR": SVR(),
            "XGBoost": xgb.XGBRegressor(),
            "LightGBM": lgb.LGBMRegressor()
        }

    selected_models = st.multiselect("Select Models to Train", list(models.keys()), default=list(models.keys())[:1])

    if st.button("Train Models"):
        trained_pipes = {}
        results = {}
        
        for name in selected_models:
            model = models[name]
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            if task_type == "classification":
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                try:
                    auc = roc_auc_score(y_test, pipe.predict_proba(X_test), multi_class="ovr")
                except:
                    auc = None
                results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC AUC": auc}

            else:
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                results[name] = {"R²": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}

            trained_pipes[name] = pipe

        # Store models in session state instead of downloading
        st.session_state.trained_models = trained_pipes
        st.session_state.model_results = results
        
        # Create results display
        results_df = pd.DataFrame(results).T
        styled_df = results_df.style.highlight_max(axis=0, color="lightgreen")
        st.dataframe(styled_df)
        
        # Determine best model and store it
        if task_type == "classification":
            best_model_name = results_df["F1"].idxmax()
        else:
            best_model_name = results_df["R²"].idxmax()
            
        st.session_state.best_model = {
            "name": best_model_name,
            "pipeline": trained_pipes[best_model_name],
            "metrics": results_df.loc[best_model_name].to_dict()
        }
        
        # Show success message but don't download automatically
        st.success("Models trained successfully! Go to the Export page to download them.")
        
        # Show best model analysis
        st.markdown("### Best Model Analysis")
        st.write(f"Best model: **{best_model_name}**")
        st.json(st.session_state.best_model["metrics"])

##################################
# Page 5: Training
#################################
# ========================= pages/5_Training.py =========================
# Training Page for Hyperparameter Tuning, Cross-validation, Early Stopping with Visualizations

elif page == "Training":
    st.header("Training")
    ensure_session_state()
    inject_css()

    st.markdown("# Model Training & Hyperparameter Tuning")
    st.caption("Tune models with grid/random/optuna search, apply cross-validation, early stopping, visualize results, and save.")

    if st.session_state.df is None:
        st.warning("No dataset loaded. Please upload and split data first.")
        st.stop()

    splits = st.session_state.get("split_result")
    if not splits:
        st.warning("Please perform Train-Test Split and build pipeline first.")
        st.stop()

    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    # Detect task type properly
    def detect_task_type(y):
        """Detect if the problem is classification or regression"""
        if pd.api.types.is_numeric_dtype(y):
            unique_values = y.nunique()
            if unique_values <= 15 and all(val in range(int(unique_values)) for val in y.dropna().unique()):
                return "classification"
            else:
                return "regression"
        else:
            return "classification"

    task_type = detect_task_type(y_train)
    st.info(f"Detected Task Type: **{task_type}**")

    # Load last trained pipelines if available
    trained_pipelines = {}

    st.markdown("## Hyperparameter Tuning")
    search_type = st.radio("Search Strategy", ["Grid Search", "Random Search", "Optuna"])

    # Set appropriate default parameter grid based on task type
    if task_type == "classification":
        default_param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_split': [2, 5, 10],
            'model__class_weight': [None, 'balanced']
        }
        scoring_options = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc"]
    else:
        default_param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
        scoring_options = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error", "explained_variance"]

    param_grid = st.text_area("Parameter Grid (dict format)", json.dumps(default_param_grid, indent=2))
    cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
    scoring_choice = st.selectbox("Scoring Metric", scoring_options, index=0)

    # Early stopping toggle (only for certain models)
    use_early_stopping = st.checkbox("Use Early Stopping (for models that support it)", value=False)

    if st.button("Run Tuning"):
        try:
            param_dict = json.loads(param_grid)
            
            # Validate parameter grid matches task type
            if task_type == "regression" and any('class_weight' in key for key in param_dict.keys()):
                st.error("Error: 'class_weight' parameter is for classification only!")
                st.stop()
                
        except json.JSONDecodeError:
            st.error("Invalid JSON format for parameter grid")
            st.stop()

        # Use appropriate base model
        if task_type == "classification":
            base_model = RandomForestClassifier()
        else:
            base_model = RandomForestRegressor()

        # Get preprocessor from session state or create default
        preprocessor = st.session_state.get("last_preprocessor")
        if preprocessor is None:
            num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
            cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler())]), num_cols),
                    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
                ]
            )

        pipe = Pipeline([("preprocessor", preprocessor), ("model", base_model)])

        if search_type == "Grid Search":
            search = GridSearchCV(pipe, param_dict, cv=cv_folds, scoring=scoring_choice, 
                                 n_jobs=-1, return_train_score=True, error_score='raise')
            search.fit(X_train, y_train)
            st.success("Grid Search complete.")
            results_df = pd.DataFrame(search.cv_results_)
            st.dataframe(results_df[["params", "mean_test_score", "mean_train_score"]])

            # Heatmap if two hyperparameters
            if len(param_dict.keys()) == 2:
                keys = list(param_dict.keys())
                pivot = results_df.pivot(index=f"param_{keys[0]}", columns=f"param_{keys[1]}", values="mean_test_score")
                fig, ax = plt.subplots()
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", ax=ax)
                st.pyplot(fig)

        elif search_type == "Random Search":
            search = RandomizedSearchCV(pipe, param_dict, cv=cv_folds, n_iter=10, 
                                       scoring=scoring_choice, n_jobs=-1, return_train_score=True)
            search.fit(X_train, y_train)
            st.success("Random Search complete.")
            results_df = pd.DataFrame(search.cv_results_)
            st.dataframe(results_df[["params", "mean_test_score", "mean_train_score"]])

            # Lineplot of mean test scores
            fig, ax = plt.subplots()
            sns.lineplot(x=range(len(results_df)), y="mean_test_score", data=results_df, marker="o", ax=ax)
            st.pyplot(fig)

        else:  # Optuna
            def objective(trial):
                n_estimators = trial.suggest_int("model__n_estimators", 50, 200)
                max_depth = trial.suggest_int("model__max_depth", 2, 20)
                
                if task_type == "classification":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                else:
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                    
                pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
                return cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=scoring_choice).mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=15)
            st.write("Best params:", study.best_params)
            st.write("Best score:", study.best_value)

            # Optuna visualizations
            st.plotly_chart(plot_optimization_history(study))
            st.plotly_chart(plot_param_importances(study))

        if search_type in ["Grid Search", "Random Search"]:
            st.write("Best Parameters:", search.best_params_)
            st.write("Best CV Score:", search.best_score_)

            # Save tuned model
            joblib.dump(search.best_estimator_, "best_model_pipeline.joblib")
            st.success("Best model pipeline saved as joblib.")

    st.markdown("---")
    download_panel(st.session_state.df, filename_basename=st.session_state.get("dataset_name") or "dataset")
    st.warning("**Reminder:** Save your progress by downloading the dataset before closing. Do not rename the file if you plan to reload it later.")


# elif page == "Training":
#     st.header("Training")
#     ensure_session_state()
#     inject_css()

#     st.markdown("# Model Training & Hyperparameter Tuning")
#     st.caption("Tune models with grid/random/optuna search, apply cross-validation, early stopping, visualize results, and save.")

#     if st.session_state.df is None:
#         st.warning("No dataset loaded. Please upload and split data first.")
#         st.stop()

#     splits = st.session_state.get("split_result")
#     if not splits:
#         st.warning("Please perform Train-Test Split and build pipeline first.")
#         st.stop()

#     X_train, X_test = splits["X_train"], splits["X_test"]
#     y_train, y_test = splits["y_train"], splits["y_test"]

#     # Load last trained pipelines if available
#     trained_pipelines = {}

#     st.markdown("## Hyperparameter Tuning")
#     search_type = st.radio("Search Strategy", ["Grid Search", "Random Search", "Optuna"])

#     param_grid = st.text_area("Parameter Grid (dict format)", "{\n    'model__n_estimators': [50, 100],\n    'model__max_depth': [3, 5, None]\n}")

#     cv_folds = st.slider("Cross-validation folds", 2, 10, 5)

#     # Early stopping toggle
#     use_early_stopping = st.checkbox("Use Early Stopping (for models that support it)", value=False)

#     # Custom scoring metric selection
#     scoring_choice = st.selectbox("Scoring Metric", [
#         "accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "r2", "neg_mean_squared_error"
#     ], index=0)

#     if st.button("Run Tuning"):
#         try:
#             import ast
#             grid = ast.literal_eval(param_grid)
#         except Exception as e:
#             st.error(f"Invalid parameter grid: {e}")
#             st.stop()

#         from sklearn.ensemble import RandomForestClassifier
#         from sklearn.pipeline import Pipeline
#         # Demo: placeholder, in practice use selected model & preprocessor from session_state
#         preprocessor = st.session_state.get("last_preprocessor")

#         if preprocessor is None:
#             num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
#             cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
#                                     ("scaler", StandardScaler())]), num_cols),
#                     ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
#                                     ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
#                 ]
#             )

#         model = RandomForestClassifier()
#         pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

#         if search_type == "Grid Search":
#             search = GridSearchCV(pipe, grid, cv=cv_folds, scoring=scoring_choice, n_jobs=-1, return_train_score=True)
#             search.fit(X_train, y_train)
#             st.success("Grid Search complete.")
#             results_df = pd.DataFrame(search.cv_results_)
#             st.dataframe(results_df[["params", "mean_test_score", "mean_train_score"]])

#             # Heatmap if two hyperparameters
#             if len(grid.keys()) == 2:
#                 keys = list(grid.keys())
#                 pivot = results_df.pivot(index=f"param_{keys[0]}", columns=f"param_{keys[1]}", values="mean_test_score")
#                 fig, ax = plt.subplots()
#                 sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", ax=ax)
#                 st.pyplot(fig)

#         elif search_type == "Random Search":
#             search = RandomizedSearchCV(pipe, grid, cv=cv_folds, n_iter=10, scoring=scoring_choice, n_jobs=-1, return_train_score=True)
#             search.fit(X_train, y_train)
#             st.success("Random Search complete.")
#             results_df = pd.DataFrame(search.cv_results_)
#             st.dataframe(results_df[["params", "mean_test_score", "mean_train_score"]])

#             # Lineplot of mean test scores
#             fig, ax = plt.subplots()
#             sns.lineplot(x=range(len(results_df)), y="mean_test_score", data=results_df, marker="o", ax=ax)
#             st.pyplot(fig)

#         else:  # Optuna
#             def objective(trial):
#                 n_estimators = trial.suggest_int("model__n_estimators", 50, 200)
#                 max_depth = trial.suggest_int("model__max_depth", 2, 20)
#                 model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
#                 pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
#                 return cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=scoring_choice).mean()

#             study = optuna.create_study(direction="maximize")
#             study.optimize(objective, n_trials=15)
#             st.write("Best params:", study.best_params)
#             st.write("Best score:", study.best_value)

#             # Optuna visualizations
#             st.plotly_chart(plot_optimization_history(study))
#             st.plotly_chart(plot_param_importances(study))

#         if search_type in ["Grid Search", "Random Search"]:
#             st.write("Best Parameters:", search.best_params_)
#             st.write("Best CV Score:", search.best_score_)

#             # Save tuned model
#             joblib.dump(search.best_estimator_, "best_model_pipeline.joblib")
#             st.success("Best model pipeline saved as joblib.")

#     st.markdown("---")

#     download_panel(st.session_state.df, filename_basename=st.session_state.get("dataset_name") or "dataset")
#     st.warning("**Reminder:** Save your progress by downloading the dataset before closing. Do not rename the file if you plan to reload it later.")

#########################################
# Page 6: Final Evaluation
#########################################

# ========================= pages/6_Final_Evaluation.py =========================
# Final Evaluation Page for assessing trained models with full metrics, comparisons, and reports

elif page == "Final Evaluation":
    st.header("Final Evaluation")

    ensure_session_state()
    inject_css()

    st.markdown("# Final Evaluation")
    st.caption("Evaluate trained models with metrics, plots, SHAP, comparisons, and downloadable reports.")

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
            models_to_compare[name] = joblib.load(name)
        except Exception:
            continue

    if not models_to_compare:
        st.warning("No trained models found. Please train and save models first.")
        st.stop()

    results_summary = {}

    for label, model in models_to_compare.items():
        start = time.time()
        y_pred = model.predict(X_test)
        runtime = time.time() - start

        # Detect classification vs regression
        task_type = "classification" if len(np.unique(y_train)) < 20 and y_train.dtype != float else "regression"

        if task_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")
            results_summary[label] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "Runtime (s)": runtime}

            st.markdown(f"## Classification Metrics — {label}")
            st.write(results_summary[label])

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # ROC / PR curves
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                if y_prob.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                    auc = roc_auc_score(y_test, y_prob[:, 1])
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                    ax.plot([0, 1], [0, 1], linestyle="--")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()
                    st.pyplot(fig)

                    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob[:, 1])
                    fig, ax = plt.subplots()
                    ax.plot(rec_curve, prec_curve, label="Precision-Recall")
                    ax.set_xlabel("Recall")
                    ax.set_ylabel("Precision")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("Multi-class ROC not yet implemented.")

            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            #rmse = np.sqrt(mse)
            results_summary[label] = {"R²": r2, "MAE": mae, "MSE": mse,  "Runtime (s)": runtime} #"RMSE": rmse,

            st.markdown(f"## Regression Metrics — {label}")
            st.write(results_summary[label])

            # Residual plot
            residuals = y_test - y_pred
            fig, ax = plt.subplots()
            ax.scatter(y_pred, residuals)
            ax.axhline(0, linestyle="--", color="red")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            st.pyplot(fig)

            # Prediction vs Actual plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

    # Comparison table
    st.markdown("## Model Comparison")
    st.write("Visit the Prediction page for a comprehensive model comparison.")
    #st.dataframe(pd.DataFrame(results_summary).T)

    # SHAP values for the last model
    st.markdown("## Explainability (SHAP)")
    try:
        last_model = list(models_to_compare.values())[-1]
        explainer = shap.Explainer(last_model.named_steps["model"], X_test)
        shap_values = explainer(X_test)
        st.set_option("deprecation.showPyplotGlobalUse", False)
        st.pyplot(shap.summary_plot(shap_values, X_test))
    except Exception:
        st.info("SHAP not available for this model.")

    # Download evaluation report
    st.markdown("## Download Evaluation Report")
    eval_report = pd.DataFrame(results_summary).T
    eval_report.to_csv("evaluation_report.csv")
    st.download_button("Download CSV Report", data=eval_report.to_csv().encode("utf-8"), file_name="evaluation_report.csv", mime="text/csv")

    st.markdown("---")

    # Download options
    download_panel(st.session_state.df, filename_basename=st.session_state.get("dataset_name") or "dataset")
    st.warning("**Reminder:** Save your progress by downloading the dataset before closing. Do not rename the file if you plan to reload it later.")

###############################
# Page 7 : Download
###############################

# ========================= pages/7_Download_Export.py =========================
# Download / Export Page – bundle models, pipelines, datasets, reports, and metadata

# elif page == "Download / Export":
#     st.header("Download / Export")

#     ensure_session_state()
#     inject_css()

#     st.markdown("# Download / Export")
#     st.caption("Export models, pipelines, datasets, visualizations, reports, and reproducible bundles with metadata.")

#     if st.session_state.df is None:
#         st.warning("No dataset loaded. Please upload data first.")
#         st.stop()

#     # --- Discovery helpers --------------------------------------------------------
#     @st.cache_data(show_spinner=False)
#     def discover_artifacts():
#         files = {}
#         files["Models/Pipelines"] = sorted(glob.glob("*.joblib") + glob.glob("*.pkl"))
#         files["Reports"] = sorted(glob.glob("*report*.csv") + glob.glob("*report*.html") + glob.glob("*profile*.html"))
#         files["Visualizations"] = sorted(glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.html"))
#         files["SHAP"] = sorted(glob.glob("*shap*.png") + glob.glob("*shap*.html"))
#         return files

#     artifacts = discover_artifacts()

#     # --- Project metadata ---------------------------------------------------------
#     with st.expander("Bundle Metadata", expanded=True):
#         colm1, colm2, colm3 = st.columns([1,1,1])
#         with colm1:
#             project_name = st.text_input("Project Name", value=st.session_state.get("dataset_name") or "ml_playground_project")
#         with colm2:
#             author = st.text_input("Author", value="user")
#         with colm3:
#             version = st.text_input("Version", value=datetime.now().strftime("%Y.%m.%d.%H%M"))

#         notes = st.text_area("Release Notes / Comments", value="")

#     # --- Dataset exports ----------------------------------------------------------
#     st.markdown("## Dataset Exports")
#     left, right = st.columns(2)
#     with left:
#         download_panel(st.session_state.df, filename_basename=st.session_state.get("dataset_name") or "dataset")
#     with right:
#         splits = st.session_state.get("splits")
#         if splits:
#             st.markdown("**Train/Test/Val Splits**")
#             for key in ["X_train","y_train","X_val","y_val","X_test","y_test"]:
#                 if key in splits and splits[key] is not None:
#                     obj = splits[key]
#                     try:
#                         csv_bytes = obj.to_csv(index=False).encode("utf-8")
#                     except Exception:
#                         # y can be Series
#                         csv_bytes = pd.DataFrame(obj).to_csv(index=False).encode("utf-8")
#                     st.download_button(
#                         f"Download {key}.csv",
#                         data=csv_bytes,
#                         file_name=f"{project_name}_{key}.csv",
#                         mime="text/csv",
#                         key=f"dl_{key}"
#                     )
#         else:
#             st.info("No saved splits found. Create them in ✂️ Train-Test Split page.")

#     st.markdown("---")

#     # --- Select artifacts to bundle ----------------------------------------------
#     st.markdown("## Select Artifacts to Include in Bundle")
#     selected_files = []
#     for section, files in artifacts.items():
#         if not files:
#             continue
#         st.markdown(f"**{section}**")
#         cols = st.columns(3)
#         for i, f in enumerate(files):
#             with cols[i % 3]:
#                 if st.checkbox(f, key=f"art_{f}"):
#                     selected_files.append(f)

#     # Allow user to add arbitrary files
#     st.markdown("**Add other files by name (comma-separated, will include if they exist in working dir):**")
#     other_files = st.text_input("Other file names", value="")
#     if other_files.strip():
#         for f in [x.strip() for x in other_files.split(",") if x.strip()]:
#             if os.path.exists(f):
#                 selected_files.append(f)
#             else:
#                 st.warning(f"File not found and skipped: {f}")

#     # --- Build metadata / manifest -----------------------------------------------
#     meta = {
#         "project_name": project_name,
#         "author": author,
#         "version": version,
#         "timestamp": datetime.utcnow().isoformat() + "Z",
#         "platform": {
#             "python_version": platform.python_version(),
#             "system": platform.system(),
#             "release": platform.release(),
#         },
#         "dataset": {
#             "name": st.session_state.get("dataset_name") or "dataset",
#             "shape": list(st.session_state.df.shape),
#             "columns": list(st.session_state.df.columns),
#         },
#         "artifacts": selected_files,
#         "notes": notes,
#     }

#     # Installed packages snapshot
#     requirements_txt = ""
#     if pkg_resources is not None:
#         try:
#             deps = sorted([f"{d.project_name}=={d.version}" for d in pkg_resources.working_set])
#             meta["dependencies"] = deps
#             requirements_txt = "\n".join(deps)
#         except Exception:
#             meta["dependencies"] = []
#     else:
#         meta["dependencies"] = []

#     # --- Create ZIP bundle --------------------------------------------------------
#     st.markdown("## Create Downloadable Bundle (.zip)")
#     zip_name = st.text_input("Bundle file name", value=f"{project_name}_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
#     include_dataset_csv = st.checkbox("Include current dataset CSV", value=True)
#     include_session_manifest = st.checkbox("Include session manifest (splits sizes, target guess)", value=True)
#     include_requirements = st.checkbox("Include requirements.txt", value=True)

#     if st.button("Build Bundle"):
#         buffer = io.BytesIO()
#         with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
#             # Add selected artifacts
#             for path in selected_files:
#                 try:
#                     zf.write(path, arcname=os.path.join("artifacts", os.path.basename(path)))
#                 except Exception as e:
#                     st.error(f"Failed to add {path}: {e}")

#             # Add dataset
#             if include_dataset_csv:
#                 csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
#                 zf.writestr(f"data/{project_name}.csv", csv_bytes)

#             # Add splits manifest
#             if include_session_manifest:
#                 splits = st.session_state.get("splits")
#                 manifest = {"splits": {}}
#                 if splits:
#                     for key in ["X_train","y_train","X_val","y_val","X_test","y_test"]:
#                         if key in splits and splits[key] is not None:
#                             try:
#                                 shape = list(splits[key].shape)
#                             except Exception:
#                                 shape = [len(splits[key])]
#                             manifest["splits"][key] = {"shape": shape}
#                 target_guess = None
#                 if splits and "y_train" in splits:
#                     target_guess = getattr(splits["y_train"], "name", None)
#                 manifest["target_guess"] = target_guess
#                 zf.writestr("manifest.json", json.dumps(manifest, indent=2))

#             # Add metadata
#             zf.writestr("metadata.json", json.dumps(meta, indent=2))

#             # Add requirements.txt
#             if include_requirements and requirements_txt:
#                 zf.writestr("requirements.txt", requirements_txt)

#         buffer.seek(0)
#         st.download_button(
#             "⬇️ Download Bundle (.zip)",
#             data=buffer.getvalue(),
#             file_name=zip_name,
#             mime="application/zip",
#         )

#     st.markdown("---")

#     st.warning("**Reminder:** Save your progress by downloading the dataset before closing. Do not rename the file if you plan to reload it later.")

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
        include_data = st.checkbox("Include dataset", value=st.session_state.export_settings['include_data'])
        include_splits = st.checkbox("Include train/test splits", value=st.session_state.export_settings['include_splits'])
        include_reports = st.checkbox("Include reports", value=st.session_state.export_settings['include_reports'])
        include_requirements = st.checkbox("Include requirements.txt", value=st.session_state.export_settings['include_requirements'])
        
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
    
    if st.button("Create Complete Project Export"):
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
    if st.checkbox("Show all available model files"):
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

                if st.button("Predict Single"):
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

                if st.button("Predict Batch"):
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