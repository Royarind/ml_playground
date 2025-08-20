# Created by @Arindam Roy 08-2025

from __future__ import annotations
import io
from typing import Optional, Tuple
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as le
#from clean_automl_page import clean_and_automl_page

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    r2_score, mean_absolute_error, mean_squared_error
)
import xgboost as xgb
import lightgbm as lgb
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import glob
import zipfile
import json
import platform
from datetime import datetime

try:
    import pkg_resources
except Exception:
    pkg_resources = None
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, classification_report,
    r2_score, mean_absolute_error, mean_squared_error
)
import shap
import re
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score, r2_score
import optuna
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
st.set_page_config(page_title="ML Playground", layout="wide", page_icon="🤖")

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
        #"Clean + AutoML"
    ]
)

def ensure_session_state():
    """Initialize expected keys in st.session_state."""
    ss = st.session_state
    ss.setdefault("df", None)                 # current working dataset (canonical in-session)
    ss.setdefault("versions", [])             # list of (name, df) tuples for quick reverts
    ss.setdefault("dataset_name", None)       # friendly name for current df
    ss.setdefault("notices", [])              # transient notices


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
        # Try utf-8, fallback to latin-1
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
    """Reusable upload panel: returns a DataFrame if a new dataset is chosen.

    Supports single/multiple CSV/Excel uploads and sample datasets on Home.
    Other sources (URL, Google Sheets, SQL, Kaggle) will be exposed on Data Loading page.
    """
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

# ========================= Home.py =========================
# Landing page for the ML Playground. Save this as: Home.py
###########################################################

if page == "Home":
    st.title("ML Playground - Step by Step Guide")
    st.write("Welcome to the Machine Learning Playground! Follow these steps to build, train, and deploy your ML models.")
    
    ensure_session_state()
    inject_css()

    # Step-by-step guide
    st.markdown("##Step-by-Step Workflow")
    
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
    st.markdown("##Current Project Status")
    
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
    st.markdown("##Quick Actions")
    
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

# ========================= pages/1_Data_Loading.py =========================
# Data Loading Page for ML Playground
# Save this as: pages/1_Data_Loading.py

elif page == "Data Loading":
    st.header("Data Loading")
    st.markdown("# Data Loading")
    st.caption("Load your dataset from multiple sources — file, preloaded, URL, SQL, Google Sheets, or Kaggle")

    ensure_session_state()
    inject_css()

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
        with tabs[1]:
            left_on = st.selectbox("Left key", [None] + df.columns.tolist())
            right_on = st.text_input("Right key (name in uploaded file)")
            how = st.selectbox("How", ["left","inner","right","outer"], index=0)
            merge_file = st.file_uploader("Upload CSV/Excel to merge with", type=["csv","xlsx"], key="merge_file")
            if merge_file and left_on and right_on and st.button("Merge now"):
                other = pd.read_csv(merge_file) if merge_file.name.endswith(".csv") else pd.read_excel(merge_file)
                st.session_state[DF_KEY] = st.session_state[DF_KEY].merge(other, left_on=left_on, right_on=right_on, how=how)
                st.success("Merged into current dataset.")

    # ---------- column ops ----------
    with st.expander("Column Operations", expanded=False):
        cols = df.columns.tolist()
        c1, c2 = st.columns([1,1])
        with c1:
            drop_cols = st.multiselect("Drop columns", cols)
            if st.button("Apply drop", key="drop_cols") and drop_cols:
                st.session_state[DF_KEY] = st.session_state[DF_KEY].drop(columns=drop_cols)
                st.success(f"Dropped: {drop_cols}")
        with c2:
            rn = st.text_input("Rename (old:new, comma-separated)", "")
            if st.button("Apply rename") and rn.strip():
                mapping = {}
                for part in rn.split(","):
                    try:
                        old, new = part.split(":")
                        mapping[old.strip()] = new.strip()
                    except Exception:
                        st.error(f"Bad pair: {part}")
                if mapping:
                    st.session_state[DF_KEY] = st.session_state[DF_KEY].rename(columns=mapping)
                    st.success(f"Renamed: {mapping}")
        st.markdown("---")
        c3, c4 = st.columns([1,1])

        with c3:
            col_tc = st.selectbox("Change dtype column", [None] + cols)
            dtype = st.selectbox("to", ["int","float","str","datetime"])
            if col_tc and st.button("Convert", key="convert"):
                try:
                    series = st.session_state[DF_KEY][col_tc]

                    if dtype == "datetime":
                        st.session_state[DF_KEY][col_tc] = pd.to_datetime(series, errors="coerce")

                    elif dtype == "int":
                        st.session_state[DF_KEY][col_tc] = pd.to_numeric(series, errors="coerce").astype("Int64")

                    elif dtype == "float":
                        st.session_state[DF_KEY][col_tc] = pd.to_numeric(series, errors="coerce")

                    elif dtype == "str":
                        st.session_state[DF_KEY][col_tc] = series.astype(str).fillna("<MISSING>")

                    st.success(f"Converted {col_tc} → {dtype}")

                except Exception as e:
                    st.error(f"Type conversion failed: {e}")

        with c4:
            query = st.text_input("Drop rows by condition (pandas.query)", "")
            if st.button("Drop rows matching condition") and query:
                try:
                    idx = st.session_state[DF_KEY].query(query).index
                    st.session_state[DF_KEY] = st.session_state[DF_KEY].drop(index=idx)
                    st.success(f"Dropped {len(idx)} rows.")
                except Exception as e:
                    st.error(f"Query failed: {e}")

    # ---------- datetime tools ----------
    with st.expander("Datetime Features", expanded=False):
        dt_col = st.selectbox("Datetime column", [None] + st.session_state[DF_KEY].select_dtypes(include=["datetime64[ns]","object"]).columns.tolist())
        if dt_col and st.button("Parse to datetime"):
            st.session_state[DF_KEY][dt_col] = pd.to_datetime(st.session_state[DF_KEY][dt_col], errors="coerce")
        if dt_col:
            add_year = st.checkbox("Add year", value=True)
            add_month = st.checkbox("Add month")
            add_day = st.checkbox("Add day")
            add_wd = st.checkbox("Add weekday")
            add_hour = st.checkbox("Add hour")
            if st.button("Extract parts"):
                s = pd.to_datetime(st.session_state[DF_KEY][dt_col], errors="coerce")
                if add_year:  st.session_state[DF_KEY][f"{dt_col}_year"] = s.dt.year
                if add_month: st.session_state[DF_KEY][f"{dt_col}_month"] = s.dt.month
                if add_day:   st.session_state[DF_KEY][f"{dt_col}_day"] = s.dt.day
                if add_wd:    st.session_state[DF_KEY][f"{dt_col}_weekday"] = s.dt.weekday
                if add_hour:  st.session_state[DF_KEY][f"{dt_col}_hour"] = s.dt.hour
                st.success("Extracted datetime parts.")

    # ---------- missing values ----------
    with st.expander("Missing Values", expanded=False):
        method = st.selectbox("Strategy", ["None","Drop rows","Fill mean","Fill median","Fill mode","Fill 0"]) 
        if method != "None" and st.button("Apply NA strategy"):
            d = st.session_state[DF_KEY]
            if method == "Drop rows":
                st.session_state[DF_KEY] = d.dropna()
            elif method == "Fill mean":
                st.session_state[DF_KEY] = d.fillna(d.mean(numeric_only=True))
            elif method == "Fill median":
                st.session_state[DF_KEY] = d.fillna(d.median(numeric_only=True))
            elif method == "Fill mode":
                st.session_state[DF_KEY] = d.fillna(d.mode().iloc[0])
            else:
                st.session_state[DF_KEY] = d.fillna(0)
            st.success(f"Applied: {method}")

    # ---------- outliers ----------
    with st.expander("Outlier Detection & Removal", expanded=False):
        d = st.session_state[DF_KEY]
        num_cols = d.select_dtypes(include=np.number).columns.tolist()
        outlier_col = st.selectbox("Column", [None] + num_cols)
        method = st.selectbox("Method", ["IQR","Z-score","MAD","IsolationForest"]) 
        preview_only = st.checkbox("Preview (do not apply)", value=True)
        if outlier_col and st.button("Detect / Remove"):
            try:
                mask = pd.Series(True, index=d.index)
                s = d[outlier_col]
                if method == "IQR":
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    mask = ~((s < (q1 - 1.5*iqr)) | (s > (q3 + 1.5*iqr)))
                elif method == "Z-score":
                    z = (s - s.mean())/s.std(ddof=0)
                    mask = z.abs() < 3
                elif method == "MAD":
                    med = s.median()
                    mad = (s - med).abs().median()
                    if mad == 0:
                        mask = pd.Series(True, index=d.index)
                    else:
                        mask = ((s - med).abs()/(1.4826*mad)) < 3.5
                else:
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    preds = iso.fit_predict(s.to_frame())
                    mask = pd.Series(preds == 1, index=d.index)

                removed = (~mask).sum()
                st.info(f"Would remove {removed} rows ({removed/len(mask):.2%}).")
                fig = px.box(d, y=outlier_col, points="outliers")
                st.plotly_chart(fig, use_container_width=True)
                if not preview_only:
                    st.session_state[DF_KEY] = d[mask]
                    st.success(f"Removed outliers: {removed} rows")
            except Exception as e:
                st.error(f"Outlier step failed: {e}")

    # ---------- groupby + deletions ----------
    with st.expander("GroupBy & Row Deletion by Groups", expanded=False):
        d = st.session_state[DF_KEY]
        cols = d.columns.tolist()

        group_cols = st.multiselect("Group by columns", cols)

        if group_cols:
            # Compute group counts safely (keep junk / NaN / mixed types)
            gb = (
                d[group_cols]
                .astype(str)       # force everything to string so junk isn’t dropped
                .fillna("<MISSING>")
                .groupby(group_cols, dropna=False)
                .size()
                .reset_index(name="count")
            )

            st.markdown("### Unique Groups and Counts")
            selections = []

            for i, row in gb.iterrows():
                group_label = ", ".join([f"{c}={row[c]}" for c in group_cols])
                count = row["count"]

                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{group_label} — {count} rows")
                with col2:
                    if st.checkbox("", key=f"delgrp_{i}"):
                        selections.append(tuple(row[c] for c in group_cols))

            # If groups selected, preview rows
            if selections:
                st.markdown("### Preview rows that will be deleted")
                kdf = d[group_cols].astype(str).apply(tuple, axis=1)
                preview = d[kdf.isin(selections)]
                st.dataframe(preview, use_container_width=True)

                # Confirm deletion
                if st.button("Confirm Delete Selected Groups"):
                    keep_mask = ~kdf.isin(selections)
                    removed = (~keep_mask).sum()
                    st.session_state[DF_KEY] = d[keep_mask]
                    st.success(f"Deleted {removed} rows from {len(selections)} group(s).")



    # ---------- text cleaning ----------
    with st.expander("Text Cleaning", expanded=False):
        d = st.session_state[DF_KEY]
        text_cols = d.select_dtypes(include="object").columns.tolist()
        if not text_cols:
            st.info("No object/text columns detected.")
        else:
            col = st.selectbox("Text column", text_cols)
            actions = st.multiselect("Choose actions", [
                "lowercase","uppercase","remove_digits","remove_punct","strip_spaces",
                "remove_phrases","remove_firstN","remove_lastN","keep_firstN","keep_lastN",
                "min_token_len","stopwords","stemming","lemmatize"
            ], default=["lowercase","strip_spaces"])
            phrases = st.text_input("Phrases to remove (space-separated)", "")
            N = st.number_input("N (for first/last ops)", min_value=0, value=0, step=1)
            minlen = st.number_input("Min token length", min_value=0, value=0, step=1)
            if st.button("Apply text cleaning"):
                s = d[col].astype(str)
                if "lowercase" in actions: s = s.str.lower()
                if "uppercase" in actions: s = s.str.upper()
                if "remove_digits" in actions: s = s.str.replace(r"\d+", "", regex=True)
                if "remove_punct" in actions: s = s.str.replace(r"[\p{Punct}]", "", regex=True)
                if "strip_spaces" in actions: s = s.str.strip()
                if "remove_phrases" in actions and phrases:
                    for ph in phrases.split():
                        s = s.str.replace(re.escape(ph), "", regex=True)
                if "remove_firstN" in actions and N>0:
                    s = s.str.replace(rf"^(?:\S+\s+){{{N}}}", "", regex=True)
                if "remove_lastN" in actions and N>0:
                    s = s.str.replace(rf"(?:\s+\S+){{{N}}}$", "", regex=True)
                if "keep_firstN" in actions and N>0:
                    s = s.str.replace(rf"^((?:\S+\s+){{0,{N}}}\S*).*", r"\1", regex=True)
                if "keep_lastN" in actions and N>0:
                    s = s.str.replace(rf".*(?:\s+((?:\S+\s+){{0,{N-1}}}\S+))$", r"\1", regex=True) if N>0 else s
                if "min_token_len" in actions and minlen>0:
                    s = s.apply(lambda t: " ".join([w for w in t.split() if len(w)>=minlen]))
                if "stopwords" in actions and nltk is not None:
                    try:
                        sw = set(stopwords.words("english"))
                    except Exception:
                        sw = set()
                    s = s.apply(lambda t: " ".join([w for w in t.split() if w.lower() not in sw]))
                if "stemming" in actions and SnowballStemmer is not None:
                    stemmer = SnowballStemmer("english")
                    s = s.apply(lambda t: " ".join(stemmer.stem(w) for w in t.split()))
                if "lemmatize" in actions and WordNetLemmatizer is not None:
                    lemm = WordNetLemmatizer()
                    s = s.apply(lambda t: " ".join(lemm.lemmatize(w) for w in t.split()))
                st.session_state[DF_KEY][col] = s
                st.success("Applied text cleaning.")

# Initialize session state for code if it doesn't exist
    if 'custom_code' not in st.session_state:
        st.session_state.custom_code = (
            "# df is a pandas DataFrame\n"
            "# Example: create a new column\n"
            "df['total'] = df.select_dtypes(include='number').sum(axis=1)\n"
            "result = df\n"
    )

    with st.expander("Custom Python (advanced)", expanded=False):
            st.markdown(
            "Write Python that transforms the DataFrame named **df**. "
            "Set `result = df` to return your new DataFrame. "
            "Nothing is applied until you click **Apply**."
        )

        # Use the session state value instead of a fixed default
    code = st.text_area(
        "Code",
        value=st.session_state.custom_code,
        height=200,
        key="code_editor"  # Add a unique key
    )
        
    # Update session state when code changes
    if code != st.session_state.custom_code:
        st.session_state.custom_code = code

    dry = st.checkbox("Dry-run preview (no apply)", value=True)

    if st.button("Run code"):
    # Safe namespace
        env = {
            "np": np,
            "pd": pd,
            "df": st.session_state[DF_KEY].copy(),
            "result": None,
        }
        # restrict builtins
        safe_builtin_names = ("abs", "min", "max", "len", "range", "sum",
                            "enumerate", "zip", "sorted", "map", "filter", "all", "any")
        builtins_dict = getattr(__builtins__, "__dict__", __builtins__)
        safe_builtins = {k: builtins_dict[k] for k in safe_builtin_names}

        try:
            exec(compile(code, "<user_code>", "exec"), {"__builtins__": safe_builtins}, env)
            res = env.get("result")

            if isinstance(res, pd.DataFrame):
                before_cols = set(st.session_state[DF_KEY].columns)
                after_cols = set(res.columns)

                st.info("🔍 Differences from current DataFrame:")
                st.json({
                    "rows_before": len(st.session_state[DF_KEY]),
                    "rows_after": len(res),
                    "added_cols": sorted(list(after_cols - before_cols)),
                    "dropped_cols": sorted(list(before_cols - after_cols)),
                })

                st.dataframe(res.head(), use_container_width=True)

                if not dry and st.button("Apply result to session"):
                    st.session_state[DF_KEY] = res
                    st.success("Applied custom transform.")

                    # keep code history
                    st.session_state.setdefault("_code_history", []).append(code)

            else:
                st.warning("Your code must set `result = <DataFrame>`. Nothing changed.")
                st.write(f"Got object of type: {type(res)}")

        except Exception as e:
            st.error(f"Execution failed: {e}")


    # ---------- visualizations (customisable + before/after) ----------
    with st.expander("Visualizations (customisable, 2D/3D/Animated)", expanded=False):
        d = st.session_state[DF_KEY]
        cols = d.columns.tolist()
        left, right = st.columns([1,2])
        with left:
            kind = st.selectbox("Chart type", [
                "Histogram","Box","Violin","Scatter","Line","Area","Pie","Density 2D","Heatmap",
                "3D Scatter","3D Surface","Animated Scatter","Animated Bar"
            ])
            x = st.selectbox("X", [None] + cols)
            y = st.selectbox("Y", [None] + cols)
            z = st.selectbox("Z (for 3D)", [None] + cols)
            color = st.selectbox("Color", [None] + cols)
            size = st.selectbox("Size", [None] + cols)
            symbol = st.selectbox("Symbol", [None] + cols)
            facet_row = st.selectbox("Facet row", [None] + cols)
            facet_col = st.selectbox("Facet col", [None] + cols)
            marginal_x = st.selectbox("Marginal X", [None, "histogram","violin","box","rug"], index=0)
            marginal_y = st.selectbox("Marginal Y", [None, "histogram","violin","box","rug"], index=0)
            trendline = st.selectbox("Trendline", [None, "ols", "lowess"], index=0 if sm is None else 1)
            animation_frame = st.selectbox("Animation frame", [None] + cols)
            template = st.selectbox("Theme", ["plotly","plotly_white","plotly_dark","ggplot2","seaborn","simple_white"])    
            title = st.text_input("Title", "")
            use_snapshot = st.checkbox("Show Before/After (use snapshot as Before)")
        with right:
            if st.button("Render chart"):
                try:
                    fig_after = make_chart(d, kind, x, y, z, color, size, symbol,
                                        facet_row, facet_col, marginal_x, marginal_y,
                                        trendline, animation_frame, template, title)
                    if use_snapshot and st.session_state.eda_snapshot is not None:
                        label, snap = st.session_state.eda_snapshot
                        t1, t2 = st.tabs([f"Before — {label}", "After (current)"])
                        with t1:
                            fig_before = make_chart(snap, kind, x, y, z, color, size, symbol,
                                                    facet_row, facet_col, marginal_x, marginal_y,
                                                    trendline, animation_frame, template, f"Before: {title}")
                            st.plotly_chart(fig_before, use_container_width=True)
                        with t2:
                            st.plotly_chart(fig_after, use_container_width=True)
                    else:
                        st.plotly_chart(fig_after, use_container_width=True)
                except Exception as e:
                    st.error(f"Plot failed: {e}")

        # ---------- automated profiling ----------
    with st.expander("Automated Profiling (YData)", expanded=False):
        if ProfileReport is None:
            st.info("ydata-profiling not installed in this environment.")
        else:
            run = st.checkbox("Run profiling (can take time)")
            if run:
                try:
                    profile = ProfileReport(st.session_state[DF_KEY], title="YData Profiling Report", explorative=True)
                    html = profile.to_html()
                    if components is not None:
                        components.html(html, height=900, scrolling=True)
                    # download button
                    buf = io.BytesIO(html.encode("utf-8"))
                    st.download_button("Download HTML report", data=buf.getvalue(), file_name="eda_profile_report.html", mime="text/html")
                except Exception as e:
                    st.error(f"Profiling failed: {e}")

    st.markdown("---")
    download_panel(st.session_state[DF_KEY], filename_basename=st.session_state.get("dataset_name") or "dataset")
    st.warning("**Reminder:** Save your progress by downloading the dataset before closing. Do not rename the file if you plan to reload it later.")

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

    trained_pipes = {}
    results = {}

    if st.button("Train Models"):
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

        # Create results_df FIRST before using it
        results_df = pd.DataFrame(results).T
        styled_df = results_df.style.highlight_max(axis=0, color="lightgreen")
        st.dataframe(styled_df)

        # Save ALL trained models with descriptive filenames
        for name, pipe in trained_pipes.items():
            # Create safe filename (remove special characters)
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            filename = f"model_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            
            joblib.dump({
                "pipeline": pipe,
                "metrics": results[name],
                "task_type": task_type,
                "classes": y_train.unique().tolist() if task_type=="classification" else None,
                "model_name": name,
                "training_date": datetime.now().isoformat()
            }, filename)
            
            st.success(f"Saved {name} as {filename}")

        # Also save the best model separately for convenience
        if task_type == "classification":
            best_model_name = results_df["F1"].idxmax()
        else:
            best_model_name = results_df["R²"].idxmax()

        best_pipe = trained_pipes[best_model_name]
        best_metrics = results_df.loc[best_model_name].to_dict()
        
        joblib.dump({
            "pipeline": best_pipe,
            "metrics": best_metrics,
            "task_type": task_type,
            "classes": y_train.unique().tolist() if task_type=="classification" else None,
            "model_name": f"BEST_{best_model_name}",
            "training_date": datetime.now().isoformat()
        }, "best_model_pipeline.joblib")
        
        st.success(f"Best model ({best_model_name}) saved as best_model_pipeline.joblib")
        
        # Show best model analysis
        st.markdown("### Best Model Analysis")
        st.write(f"Best model: **{best_model_name}**")
        st.json(best_metrics)


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

    # Load last trained pipelines if available
    trained_pipelines = {}

    st.markdown("## Hyperparameter Tuning")
    search_type = st.radio("Search Strategy", ["Grid Search", "Random Search", "Optuna"])

    param_grid = st.text_area("Parameter Grid (dict format)", "{\n    'model__n_estimators': [50, 100],\n    'model__max_depth': [3, 5, None]\n}")

    cv_folds = st.slider("Cross-validation folds", 2, 10, 5)

    # Early stopping toggle
    use_early_stopping = st.checkbox("Use Early Stopping (for models that support it)", value=False)

    # Custom scoring metric selection
    scoring_choice = st.selectbox("Scoring Metric", [
        "accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "r2", "neg_mean_squared_error"
    ], index=0)

    if st.button("Run Tuning"):
        try:
            import ast
            grid = ast.literal_eval(param_grid)
        except Exception as e:
            st.error(f"Invalid parameter grid: {e}")
            st.stop()

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        # Demo: placeholder, in practice use selected model & preprocessor from session_state
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

        model = RandomForestClassifier()
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        if search_type == "Grid Search":
            search = GridSearchCV(pipe, grid, cv=cv_folds, scoring=scoring_choice, n_jobs=-1, return_train_score=True)
            search.fit(X_train, y_train)
            st.success("Grid Search complete.")
            results_df = pd.DataFrame(search.cv_results_)
            st.dataframe(results_df[["params", "mean_test_score", "mean_train_score"]])

            # Heatmap if two hyperparameters
            if len(grid.keys()) == 2:
                keys = list(grid.keys())
                pivot = results_df.pivot(index=f"param_{keys[0]}", columns=f"param_{keys[1]}", values="mean_test_score")
                fig, ax = plt.subplots()
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", ax=ax)
                st.pyplot(fig)

        elif search_type == "Random Search":
            search = RandomizedSearchCV(pipe, grid, cv=cv_folds, n_iter=10, scoring=scoring_choice, n_jobs=-1, return_train_score=True)
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
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
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
    st.dataframe(pd.DataFrame(results_summary).T)

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

elif page == "Download / Export":
    st.header("Download / Export")

    ensure_session_state()
    inject_css()

    st.markdown("# Download / Export")
    st.caption("Export models, pipelines, datasets, visualizations, reports, and reproducible bundles with metadata.")

    if st.session_state.df is None:
        st.warning("No dataset loaded. Please upload data first.")
        st.stop()

    # --- Discovery helpers --------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def discover_artifacts():
        files = {}
        files["Models/Pipelines"] = sorted(glob.glob("*.joblib") + glob.glob("*.pkl"))
        files["Reports"] = sorted(glob.glob("*report*.csv") + glob.glob("*report*.html") + glob.glob("*profile*.html"))
        files["Visualizations"] = sorted(glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.html"))
        files["SHAP"] = sorted(glob.glob("*shap*.png") + glob.glob("*shap*.html"))
        return files

    artifacts = discover_artifacts()

    # --- Project metadata ---------------------------------------------------------
    with st.expander("Bundle Metadata", expanded=True):
        colm1, colm2, colm3 = st.columns([1,1,1])
        with colm1:
            project_name = st.text_input("Project Name", value=st.session_state.get("dataset_name") or "ml_playground_project")
        with colm2:
            author = st.text_input("Author", value="user")
        with colm3:
            version = st.text_input("Version", value=datetime.now().strftime("%Y.%m.%d.%H%M"))

        notes = st.text_area("Release Notes / Comments", value="")

    # --- Dataset exports ----------------------------------------------------------
    st.markdown("## Dataset Exports")
    left, right = st.columns(2)
    with left:
        download_panel(st.session_state.df, filename_basename=st.session_state.get("dataset_name") or "dataset")
    with right:
        splits = st.session_state.get("splits")
        if splits:
            st.markdown("**Train/Test/Val Splits**")
            for key in ["X_train","y_train","X_val","y_val","X_test","y_test"]:
                if key in splits and splits[key] is not None:
                    obj = splits[key]
                    try:
                        csv_bytes = obj.to_csv(index=False).encode("utf-8")
                    except Exception:
                        # y can be Series
                        csv_bytes = pd.DataFrame(obj).to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"Download {key}.csv",
                        data=csv_bytes,
                        file_name=f"{project_name}_{key}.csv",
                        mime="text/csv",
                        key=f"dl_{key}"
                    )
        else:
            st.info("No saved splits found. Create them in ✂️ Train-Test Split page.")

    st.markdown("---")

    # --- Select artifacts to bundle ----------------------------------------------
    st.markdown("## Select Artifacts to Include in Bundle")
    selected_files = []
    for section, files in artifacts.items():
        if not files:
            continue
        st.markdown(f"**{section}**")
        cols = st.columns(3)
        for i, f in enumerate(files):
            with cols[i % 3]:
                if st.checkbox(f, key=f"art_{f}"):
                    selected_files.append(f)

    # Allow user to add arbitrary files
    st.markdown("**Add other files by name (comma-separated, will include if they exist in working dir):**")
    other_files = st.text_input("Other file names", value="")
    if other_files.strip():
        for f in [x.strip() for x in other_files.split(",") if x.strip()]:
            if os.path.exists(f):
                selected_files.append(f)
            else:
                st.warning(f"File not found and skipped: {f}")

    # --- Build metadata / manifest -----------------------------------------------
    meta = {
        "project_name": project_name,
        "author": author,
        "version": version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "platform": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "dataset": {
            "name": st.session_state.get("dataset_name") or "dataset",
            "shape": list(st.session_state.df.shape),
            "columns": list(st.session_state.df.columns),
        },
        "artifacts": selected_files,
        "notes": notes,
    }

    # Installed packages snapshot
    requirements_txt = ""
    if pkg_resources is not None:
        try:
            deps = sorted([f"{d.project_name}=={d.version}" for d in pkg_resources.working_set])
            meta["dependencies"] = deps
            requirements_txt = "\n".join(deps)
        except Exception:
            meta["dependencies"] = []
    else:
        meta["dependencies"] = []

    # --- Create ZIP bundle --------------------------------------------------------
    st.markdown("## Create Downloadable Bundle (.zip)")
    zip_name = st.text_input("Bundle file name", value=f"{project_name}_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    include_dataset_csv = st.checkbox("Include current dataset CSV", value=True)
    include_session_manifest = st.checkbox("Include session manifest (splits sizes, target guess)", value=True)
    include_requirements = st.checkbox("Include requirements.txt", value=True)

    if st.button("Build Bundle"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Add selected artifacts
            for path in selected_files:
                try:
                    zf.write(path, arcname=os.path.join("artifacts", os.path.basename(path)))
                except Exception as e:
                    st.error(f"Failed to add {path}: {e}")

            # Add dataset
            if include_dataset_csv:
                csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
                zf.writestr(f"data/{project_name}.csv", csv_bytes)

            # Add splits manifest
            if include_session_manifest:
                splits = st.session_state.get("splits")
                manifest = {"splits": {}}
                if splits:
                    for key in ["X_train","y_train","X_val","y_val","X_test","y_test"]:
                        if key in splits and splits[key] is not None:
                            try:
                                shape = list(splits[key].shape)
                            except Exception:
                                shape = [len(splits[key])]
                            manifest["splits"][key] = {"shape": shape}
                target_guess = None
                if splits and "y_train" in splits:
                    target_guess = getattr(splits["y_train"], "name", None)
                manifest["target_guess"] = target_guess
                zf.writestr("manifest.json", json.dumps(manifest, indent=2))

            # Add metadata
            zf.writestr("metadata.json", json.dumps(meta, indent=2))

            # Add requirements.txt
            if include_requirements and requirements_txt:
                zf.writestr("requirements.txt", requirements_txt)

        buffer.seek(0)
        st.download_button(
            "⬇️ Download Bundle (.zip)",
            data=buffer.getvalue(),
            file_name=zip_name,
            mime="application/zip",
        )

    st.markdown("---")

    st.warning("**Reminder:** Save your progress by downloading the dataset before closing. Do not rename the file if you plan to reload it later.")


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
    model_files = glob.glob("*.joblib") + glob.glob("*.pkl")
    model_files = [f for f in model_files if not f.startswith('.')]
    
    st.markdown("## Select Model")
    
    if not model_files:
        st.warning("No trained models found. Please train models on the Pipeline page first.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model_file = st.selectbox(
            "Choose a trained model",
            options=[""] + model_files,
            help="Select from previously trained models"
        )
    
    with col2:
        st.markdown("### Or upload new model")
        uploaded_model_file = st.file_uploader(
            "Upload model file", 
            type=["joblib", "pkl"],
            help="Upload a .joblib or .pkl file from your computer"
        )

    # --- Load selected model ---
    loaded_model, meta_metrics, target_classes, model_name, task_type = None, None, None, None, None
    
    if selected_model_file:
        try:
            saved_obj = joblib.load(selected_model_file)
            st.success(f"Loaded model from: {selected_model_file}")
        except Exception as e:
            st.error(f"Failed to load {selected_model_file}: {e}")
    
    elif uploaded_model_file:
        try:
            with open("temp_uploaded_model.joblib", "wb") as f:
                f.write(uploaded_model_file.getbuffer())
            saved_obj = joblib.load("temp_uploaded_model.joblib")
            st.success(f"Uploaded model loaded successfully: {uploaded_model_file.name}")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")
    
    # Extract model information
    if 'saved_obj' in locals():
        if isinstance(saved_obj, dict) and "pipeline" in saved_obj:
            loaded_model = saved_obj["pipeline"]
            meta_metrics = saved_obj.get("metrics")
            target_classes = saved_obj.get("classes")
            model_name = saved_obj.get("model_name", "Unknown Model")
            task_type = saved_obj.get("task_type")
        else:
            loaded_model = saved_obj
            model_name = "Direct Pipeline"
        
        st.info(f"Model: {model_name}")
        st.info(f"Task Type: {task_type or 'Unknown'}")
        
        if meta_metrics:
            st.markdown("### Model Performance Metrics")
            metrics_df = pd.DataFrame.from_dict(meta_metrics, orient='index', columns=['Value'])
            st.dataframe(metrics_df.style.format("{:.4f}"))

    elif page == "Prediction":
        st.header("Prediction")
        ensure_session_state()
        inject_css()

        st.markdown("# Prediction")
        st.caption("Load saved models and pipelines, perform batch or single-record inference, monitor drift.")

        # --- Model Loader ---
        # Initialize session state for current session models if not exists
        if 'current_session_models' not in st.session_state:
            st.session_state.current_session_models = []
        
        # Define current_date properly
        current_date = datetime.now().strftime('%Y%m%d')
        
        # Get only models from current session (with timestamp from today)
        current_session_models = [
            f for f in glob.glob("*.joblib") + glob.glob("*.pkl") 
            if not f.startswith('.') and current_date in f
        ]
        
        # Also include the best model if it was created today
        best_model = "best_model_pipeline.joblib"
        if os.path.exists(best_model):
            try:
                creation_time = datetime.fromtimestamp(os.path.getctime(best_model))
                if creation_time.strftime('%Y%m%d') == current_date:
                    current_session_models.append(best_model)
            except OSError:
                pass  # File might not exist or other OS error
        
        # Update session state
        st.session_state.current_session_models = current_session_models
        
        st.markdown("## Select Model")
        
        if not current_session_models:
            st.warning("No trained models found from current session. Please train models on the Pipeline page first.")
            st.info("💡 Models are only shown if they were created in the current session.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_model_file = st.selectbox(
                "Choose a trained model from current session",
                options=[""] + current_session_models,
                help="Only shows models created in the current session"
            )
        
        with col2:
            st.markdown("### Or load any model")
            uploaded_model_file = st.file_uploader(
                "Upload model file", 
                type=["joblib", "pkl"],
                help="Upload any .joblib or .pkl file"
            )
            # Show option to view all models
            if st.checkbox("Show all models (not recommended)"):
                all_models = [f for f in glob.glob("*.joblib") + glob.glob("*.pkl") if not f.startswith('.')]
                if all_models:
                    st.info("All available models:")
                    for model in all_models:
                        try:
                            creation_time = datetime.fromtimestamp(os.path.getctime(model))
                            st.write(f"• {model} (created: {creation_time.strftime('%Y-%m-%d %H:%M')})")
                        except OSError:
                            st.write(f"• {model} (creation time unavailable)")
                else:
                    st.write("No models found.")

        # --- Load selected model ---
        loaded_model, meta_metrics, target_classes, model_name, task_type = None, None, None, None, None
        
        if selected_model_file:
            try:
                saved_obj = joblib.load(selected_model_file)
                st.success(f"Loaded model from: {selected_model_file}")
                
                # Show model creation time
                try:
                    creation_time = datetime.fromtimestamp(os.path.getctime(selected_model_file))
                    st.caption(f"Model created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
                except OSError:
                    st.caption("Creation time unavailable")
                
            except Exception as e:
                st.error(f"Failed to load {selected_model_file}: {e}")
        
        elif uploaded_model_file:
            try:
                with open("temp_uploaded_model.joblib", "wb") as f:
                    f.write(uploaded_model_file.getbuffer())
                saved_obj = joblib.load("temp_uploaded_model.joblib")
                st.success(f"Uploaded model loaded successfully: {uploaded_model_file.name}")
            except Exception as e:
                st.error(f"Failed to load uploaded model: {e}")
        
        # Extract model information
        if 'saved_obj' in locals():
            if isinstance(saved_obj, dict) and "pipeline" in saved_obj:
                loaded_model = saved_obj["pipeline"]
                meta_metrics = saved_obj.get("metrics")
                target_classes = saved_obj.get("classes")
                model_name = saved_obj.get("model_name", "Unknown Model")
                task_type = saved_obj.get("task_type")
            else:
                loaded_model = saved_obj
                model_name = "Direct Pipeline"
            
            st.info(f"Model: {model_name}")
            st.info(f"Task Type: {task_type or 'Unknown'}")
            
            if meta_metrics:
                st.markdown("### Model Performance Metrics")
                metrics_df = pd.DataFrame.from_dict(meta_metrics, orient='index', columns=['Value'])
                st.dataframe(metrics_df.style.format("{:.4f}"))

        # --- Clear old models button ---
        st.markdown("---")
        st.markdown("### Model Management")
        
        if st.button("🔄 Clear All Models from Previous Sessions"):
            all_models = glob.glob("*.joblib") + glob.glob("*.pkl")
            models_deleted = 0
            for model_file in all_models:
                if not model_file.startswith('.'):
                    try:
                        # Check if model is from current session
                        creation_time = datetime.fromtimestamp(os.path.getctime(model_file))
                        if creation_time.strftime('%Y%m%d') != current_date:
                            os.remove(model_file)
                            models_deleted += 1
                    except (OSError, FileNotFoundError):
                        continue  # Skip if file doesn't exist or other OS error
            
            if models_deleted > 0:
                st.success(f"Deleted {models_deleted} models from previous sessions")
                st.rerun()
            else:
                st.info("No old models found to delete")

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
                                help=f"Select from available {col} values"
                            )
                        else:
                            input_val = st.text_input(f"{col}", value="")
                    elif pd.api.types.is_numeric_dtype(col_data):
                        # Numeric column - use number input with free input
                        # Show example value from training data as placeholder
                        example_val = float(col_data.mean()) if not col_data.empty and not pd.isna(col_data.mean()) else 0.0
                        input_val = st.number_input(
                            f"{col}",
                            value=example_val,
                            step=0.1,
                            help=f"Enter any numeric value for {col}"
                        )
                    elif pd.api.types.is_datetime64_any_dtype(col_data):
                        # Date column - use date input
                        min_date = col_data.min() if not col_data.empty else datetime.now().date()
                        max_date = col_data.max() if not col_data.empty else datetime.now().date()
                        input_val = st.date_input(f"{col}", value=min_date, min_value=min_date, max_value=max_date)
                        input_val = input_val.isoformat()
                    else:
                        # Fallback to text input
                        input_val = st.text_input(f"{col}", value="")
                    
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
            batch_file = st.file_uploader("Upload batch file (CSV/Excel)", type=["csv","xlsx"])
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
                                "⬇️ Download Predictions CSV", 
                                data=csv_bytes, 
                                file_name="predictions.csv", 
                                mime="text/csv"
                            )
                            
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")
                        st.error(f"Error details: {str(e)}")
    
    else:
        st.warning("Please load a trained model first.")

    st.markdown("---")
    st.warning("**Note:** Make sure your input data has the same features and data types as the training data.")