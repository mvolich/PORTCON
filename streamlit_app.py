import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import plotly.graph_objects as go
import plotly.express as px

# Set global font for all Plotly charts
import plotly.io as pio
pio.templates.default = "plotly_white"

# Configure global font for Plotly
pio.templates["plotly_white"].layout.font = dict(family="Ringside", size=12)
pio.templates["plotly_white"].layout.title.font = dict(family="Ringside", size=16)
from sklearn.covariance import LedoitWolf
import io
import base64
from PIL import Image
import requests
import copy
import json
import hashlib

# Page configuration
st.set_page_config(
    page_title="Portfolio Construction Model",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rubrics color palette
RUBRICS_COLORS = {
    'blue': '#001E4F',
    'medium_blue': '#2C5697', 
    'light_blue': '#7BA4DB',
    'grey': '#D8D7DF',
    'orange': '#CF4520'
}

# Custom CSS for Rubrics branding
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Ringside:wght@300;400;500;600;700&display=swap');
    
    /* Universal font application for all elements */
    html, body, [class*="css"], div, span, table, th, td, button, input, label, textarea, 
    h1, h2, h3, h4, h5, h6, p, li, ul, ol, a, strong, em, b, i, 
    .stMarkdown, .stText, .stHeader, .stSubheader, .stTitle, .stCaption,
    .stSelectbox, .stMultiselect, .stSlider, .stNumberInput, .stTextInput, 
    .stCheckbox, .stRadio, .stButton, .stFileUploader, .stDataFrame,
    .stPlotlyChart, .stAltairChart, .stLineChart, .stBarChart, .stAreaChart,
    .stExpander, .stTabs, .stColumns, .stContainer, .stSidebar,
    .stMetric, .stProgress, .stSpinner, .stBalloons, .stSnow,
    .stError, .stWarning, .stInfo, .stSuccess,
    .stJson, .stCode, .stLatex, .stMath,
    .stImage, .stVideo, .stAudio, .stDownloadButton,
    .stForm, .stFormSubmitButton, .stFormClearButton,
    .stSessionState, .stCache, .stExperimentalMemo, .stExperimentalSingleton {{
        font-family: 'Ringside', sans-serif !important;
    }}
    
    /* Specific styling for Streamlit text elements */
    .stMarkdown, .stText, .stHeader, .stSubheader, .stTitle, .stCaption {{
        font-family: 'Ringside', sans-serif !important;
        font-weight: 400 !important;
    }}
    
    /* Headers with specific weights */
    h1, .stHeader h1, .stMarkdown h1 {{
        font-family: 'Ringside', sans-serif !important;
        font-weight: 600 !important;
    }}
    
    h2, .stSubheader h2, .stMarkdown h2 {{
        font-family: 'Ringside', sans-serif !important;
        font-weight: 500 !important;
    }}
    
    h3, h4, h5, h6, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
        font-family: 'Ringside', sans-serif !important;
        font-weight: 500 !important;
    }}
    
    /* Form elements */
    .stSelectbox, .stMultiselect, .stSlider, .stNumberInput, .stTextInput, .stCheckbox, .stRadio {{
        font-family: 'Ringside', sans-serif !important;
        font-weight: 400 !important;
    }}
    
    /* Dataframes and tables - comprehensive styling */
    .stDataFrame, .stDataFrame table, .stDataFrame th, .stDataFrame td,
    .stDataFrame thead, .stDataFrame tbody, .stDataFrame tfoot,
    .stDataFrame thead th, .stDataFrame tbody td, .stDataFrame tfoot td,
    .dataframe, .dataframe table, .dataframe th, .dataframe td,
    .dataframe thead, .dataframe tbody, .dataframe tfoot,
    .dataframe thead th, .dataframe tbody td, .dataframe tfoot td,
    div[data-testid="stDataFrame"], div[data-testid="stDataFrame"] table,
    div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td,
    div[data-testid="stDataFrame"] thead, div[data-testid="stDataFrame"] tbody,
    div[data-testid="stDataFrame"] thead th, div[data-testid="stDataFrame"] tbody td {{
        font-family: 'Ringside', sans-serif !important;
        font-weight: 400 !important;
    }}
    
    /* Target specific dataframe content including styled dataframes */
    .stDataFrame .styled-table, .stDataFrame .styled-table th, .stDataFrame .styled-table td,
    .stDataFrame .pandas-styler, .stDataFrame .pandas-styler th, .stDataFrame .pandas-styler td,
    .data, .data th, .data td, .row_heading, .col_heading, .blank {{
        font-family: 'Ringside', sans-serif !important;
        font-weight: 400 !important;
    }}
    
    /* Ensure all table elements use Ringside regardless of nesting */
    table, table th, table td, table thead, table tbody, table tfoot,
    table thead th, table tbody td, table tfoot td {{
        font-family: 'Ringside', sans-serif !important;
    }}
    
    /* Specifically target Pandas Styler elements used in Portfolio Weights, Metrics, and Constraints tables */
    .stDataFrame .row0, .stDataFrame .row1, .stDataFrame .row2, .stDataFrame .row3, .stDataFrame .row4,
    .stDataFrame .row5, .stDataFrame .row6, .stDataFrame .row7, .stDataFrame .row8, .stDataFrame .row9,
    .stDataFrame .col0, .stDataFrame .col1, .stDataFrame .col2, .stDataFrame .col3, .stDataFrame .col4,
    .stDataFrame .col5, .stDataFrame .col6, .stDataFrame .col7, .stDataFrame .col8, .stDataFrame .col9,
    .stDataFrame .level0, .stDataFrame .level1, .stDataFrame .level2,
    .stDataFrame .index_name, .stDataFrame .col_heading {{
        font-family: 'Ringside', sans-serif !important;
        font-weight: 400 !important;
    }}
    
    /* Force font on all cell content */
    .stDataFrame * {{
        font-family: 'Ringside', sans-serif !important;
    }}
    
    /* Additional targeting for styled dataframes with custom backgrounds */
    [style*="background-color"] {{
        font-family: 'Ringside', sans-serif !important;
    }}
    
    /* Ensure Portfolio Weights Table, Portfolio Metrics, and Constraints Budget Usage use Ringside */
    .stDataFrame div, .stDataFrame span, .stDataFrame p, .stDataFrame text,
    .stDataFrame .ag-header-cell-text, .stDataFrame .ag-cell-value,
    .stDataFrame .ag-header-group-text, .stDataFrame .ag-header-cell-label {{
        font-family: 'Ringside', sans-serif !important;
    }}
    
    /* Target AG Grid elements if used */
    .ag-theme-streamlit, .ag-theme-streamlit .ag-header-cell-text,
    .ag-theme-streamlit .ag-cell-value, .ag-theme-streamlit .ag-row {{
        font-family: 'Ringside', sans-serif !important;
    }}
    
    /* Ensure iframe content uses Ringside if applicable */
    iframe {{
        font-family: 'Ringside', sans-serif !important;
    }}
    
    /* Buttons and interactive elements */
    .stButton, .stFileUploader, .stDownloadButton {{
        font-family: 'Ringside', sans-serif !important;
        font-weight: 500 !important;
    }}
    
    /* Sidebar specific styling */
    .stSidebar, .stSidebar * {{
        font-family: 'Ringside', sans-serif !important;
    }}
    
    /* Charts and plots - ensure text elements use Ringside */
    .js-plotly-plot, .plotly, .plotly-graph-div, .plotly-notifier {{
        font-family: 'Ringside', sans-serif !important;
    }}
    
    /* Force font loading and fallback */
    * {{
        font-family: 'Ringside', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    }}
    
    /* Ensure welcome message uses Ringside font */
    .welcome-message h2, .welcome-message p, .welcome-message ul, .welcome-message li {{
        font-family: 'Ringside', sans-serif !important;
    }}
    
    /* Ensure all Streamlit markdown content uses Ringside */
    .stMarkdown div, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown ul, .stMarkdown ol, 
    .stMarkdown li, .stMarkdown strong, .stMarkdown em, .stMarkdown b, .stMarkdown i {{
        font-family: 'Ringside', sans-serif !important;
    }}
    
    .metric-card {{
        background-color: {RUBRICS_COLORS['grey']};
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {RUBRICS_COLORS['orange']};
        margin: 0.5rem 0;
    }}
    
    .constraint-section {{
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }}
    
    .stSlider > div > div > div > div {{
        background-color: {RUBRICS_COLORS['orange']};
    }}
    
    .stButton > button {{
        background-color: {RUBRICS_COLORS['blue']};
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }}
    
    .stButton > button:hover {{
        background-color: {RUBRICS_COLORS['medium_blue']};
    }}
    
    /* Fix expander text overlap issue */
    .streamlit-expanderHeader {{
        font-family: 'Ringside', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #262730 !important;
        padding: 0.75rem 1rem !important;
        background-color: #f0f2f6 !important;
        border-radius: 0.5rem !important;
        border: 1px solid #e0e0e0 !important;
        margin-bottom: 0.5rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        background-color: #e6e9ef !important;
        border-color: #d0d0d0 !important;
    }}
    
    /* Ensure proper spacing for expander content */
    .streamlit-expanderContent {{
        padding: 1rem !important;
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-top: none !important;
        border-radius: 0 0 0.5rem 0.5rem !important;
        margin-bottom: 1rem !important;
    }}
    
    /* Fix any icon overlap issues */
    .streamlit-expanderHeader .streamlit-expanderHeaderIcon {{
        margin-right: 0.5rem !important;
        vertical-align: middle !important;
    }}
    
    /* Ensure text doesn't overlap with icons */
    .streamlit-expanderHeader span {{
        display: inline-block !important;
        vertical-align: middle !important;
        line-height: 1.4 !important;
    }}
</style>
""", unsafe_allow_html=True)

# Load Rubrics logo
@st.cache_data
def load_logo():
    try:
        response = requests.get("https://rubricsam.com/wp-content/uploads/2021/01/cropped-rubrics-logo-tight.png")
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return image
    except (requests.RequestException, IOError, Exception):
        pass
    return None

logo = load_logo()

# Header with logo moved to top right
col1, col2 = st.columns([4, 1])
with col1:
    # Empty space on the left
    st.write("")
with col2:
    if logo:
        st.image(logo, width=150)

# Sidebar for file upload and controls
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload Excel file with 'Index List' and 'Sheet2' tabs",
        type=['xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")
        
        # Load data
        xls = pd.ExcelFile(uploaded_file)
        df_raw = xls.parse('Index List')
        df_metadata_raw = xls.parse('Sheet2')
        
        st.header("⚙️ Fund Constraints")
        
        # Define rating scale
        rating_scale = {
            'AAA': 20, 'AA+': 19, 'AA': 18, 'AA-': 17,
            'A+': 16, 'A': 15, 'A-': 14,
            'BBB+': 13, 'BBB': 12, 'BBB-': 11,
            'BB+': 10, 'BB': 9, 'BB-': 8,
            'B+': 7, 'B': 6, 'B-': 5,
            'CCC+': 4, 'CCC': 3, 'CCC-': 2,
            'NR': 1
        }
        
        inverse_rating_scale = {v: k for k, v in rating_scale.items()}
        
        # Fund management - show all funds
        st.subheader("Fund Configuration")
        
        # Create tabs for each fund's constraints
        fund_tabs = st.tabs(['GFI', 'GCF', 'EYF'])
        
        # Store the fund list for later use
        fund_list = ['GFI', 'GCF', 'EYF']
        
        # Default constraints
        default_constraints = {
            'GFI': {
                'max_non_ig': 0.25,
                'max_em': 0.30,
                'max_at1': 0.15,
                'max_duration': 6.5,
                'min_rating': rating_scale['BBB-'],
                'max_hybrid': 0.15,
                'max_tbill': 0.20
            },
            'GCF': {
                'max_non_ig': 0.10,
                'max_em': 0.35,
                'max_at1': 0.10,
                'max_duration': 3.5,
                'min_rating': rating_scale['BBB'],
                'max_hybrid': 0.10,
                'max_tbill': 0.20
            },
            'EYF': {
                'max_non_ig': 1.0,
                'max_em': 1.0,
                'max_at1': 0.0001,
                'max_duration': None,
                'min_rating': rating_scale['BB'],
                'max_hybrid': 0.20,
                'max_tbill': 0.20
            }
        }
        
        # Initialize session state for constraints
        # Compute a stable hash of your defaults
        _defaults_json = json.dumps(default_constraints, sort_keys=True)
        _defaults_hash = hashlib.md5(_defaults_json.encode()).hexdigest()
        
        # If first run OR defaults have changed, reset both constraints and widget state
        if (
            'fund_constraints' not in st.session_state
            or st.session_state.get('default_constraints_hash') != _defaults_hash
        ):
            # Deep-copy so nested dicts don't share references
            st.session_state.fund_constraints = copy.deepcopy(default_constraints)
            st.session_state['default_constraints_hash'] = _defaults_hash
            
            # Remove any old slider/selectbox state so they re-init with new defaults
            for fund in default_constraints:
                st.session_state.pop(f"at1_slider_{fund}", None)
                st.session_state.pop(f"duration_slider_{fund}", None)
                st.session_state.pop(f"hybrid_slider_{fund}", None)
                st.session_state.pop(f"rating_select_{fund}", None)
        
        # Configure constraints for each fund in tabs
        for i, fund in enumerate(fund_list):
            with fund_tabs[i]:
                st.subheader(f"{fund} Constraints")
                
                constraints = st.session_state.fund_constraints[fund]
                
                # Non-editable constraints (display only)
                st.markdown("**Fixed Constraints (Cannot be changed):**")
                st.markdown(f"• Max Non-IG: {constraints['max_non_ig']*100:.1f}%")
                st.markdown(f"• Max EM: {constraints['max_em']*100:.1f}%")
                st.markdown(f"• Max T-Bills: {constraints['max_tbill']*100:.1f}%")
                
                # Editable constraints
                st.markdown("**Editable Constraints:**")
                
                # AT1 constraint
                if fund == 'EYF':
                    st.markdown("• Max AT1: 0.01% (Fixed for EYF)")
                else:
                    new_max_at1 = st.slider(
                        "Max AT1 Exposure",
                        min_value=0.0,
                        max_value=0.5,
                        value=constraints['max_at1'],
                        step=0.01,
                        help="Maximum Additional Tier 1 capital exposure",
                        key=f"at1_slider_{fund}"
                    )
                    st.session_state.fund_constraints[fund]['max_at1'] = float(new_max_at1)
                
                # Duration constraint
                if fund == 'EYF':
                    st.markdown("• Max Duration: None (Fixed for EYF)")
                else:
                    new_max_duration = st.slider(
                        "Max Duration (years)",
                        min_value=1.0,
                        max_value=10.0,
                        value=constraints['max_duration'] if constraints['max_duration'] else 5.0,
                        step=0.1,
                        help="Maximum portfolio duration",
                        key=f"duration_slider_{fund}"
                    )
                    st.session_state.fund_constraints[fund]['max_duration'] = float(new_max_duration)
                
                # Hybrid constraint
                new_max_hybrid = st.slider(
                    "Max Hybrid Exposure",
                    min_value=0.0,
                    max_value=0.5,
                    value=constraints['max_hybrid'],
                    step=0.01,
                    help="Maximum hybrid instrument exposure",
                    key=f"hybrid_slider_{fund}"
                )
                st.session_state.fund_constraints[fund]['max_hybrid'] = float(new_max_hybrid)
                
                # Rating constraint
                rating_options = list(rating_scale.keys())
                current_rating = inverse_rating_scale[constraints['min_rating']]
                new_min_rating = st.selectbox(
                    "Minimum Rating",
                    options=rating_options,
                    index=rating_options.index(current_rating),
                    help="Minimum average portfolio rating",
                    key=f"rating_select_{fund}"
                )
                st.session_state.fund_constraints[fund]['min_rating'] = rating_scale[new_min_rating]
                
                # Reset button
                if st.button(f"Reset {fund} to Defaults", key=f"reset_{fund}"):
                    st.session_state.fund_constraints[fund] = copy.deepcopy(default_constraints[fund])
                    st.rerun()

# Main content area
if uploaded_file is not None:
    # Data processing functions
    def process_data(df_raw, df_metadata_raw):
        """Process the uploaded data"""
        # Identify date and value columns
        date_columns = df_raw.columns[::2]
        value_columns = df_raw.columns[1::2]
        
        # Get common dates
        date_sets = []
        for col in date_columns:
            dates = pd.to_datetime(df_raw[col].dropna().unique())
            date_sets.append(set(dates))
        common_dates = sorted(set.intersection(*date_sets))
        
        # Clean and merge series
        series_list = []
        for date_col, value_col in zip(date_columns, value_columns):
            temp_df = df_raw[[date_col, value_col]].copy()
            temp_df.columns = ['Date', value_col]
            temp_df['Date'] = pd.to_datetime(temp_df['Date'])
            temp_df = temp_df[temp_df['Date'].isin(common_dates)]
            temp_df = temp_df.drop_duplicates(subset='Date')
            series_list.append(temp_df)
        
        df_common = series_list[0]
        for i in range(1, len(series_list)):
            df_common = pd.merge(df_common, series_list[i], on='Date', how='inner')
        
        df_common.set_index('Date', inplace=True)
        
        # Calculate returns
        df_pct_change = df_common.pct_change().dropna()
        
        # Align metadata - exactly as original file
        available_names = df_pct_change.columns.intersection(df_metadata_raw['Name'])
        df_pct_change = df_pct_change[available_names]
        df_metadata = df_metadata_raw[df_metadata_raw['Name'].isin(available_names)].set_index('Name')
        
        # Ensure correct ordering explicitly (as original file)
        df_metadata = df_metadata.loc[df_pct_change.columns]
        
        # Add this critical assertion back
        assert all(df_pct_change.columns == df_metadata.index), "Mismatch in index order"
        
        # Explicit final alignment (original file logic)
        df_metadata = df_metadata.loc[df_pct_change.columns]
        
        # Explicit numeric conversion (exactly as original file)
        df_metadata[['Is_AT1', 'Is_EM', 'Is_Non_IG', 'Is_Hybrid']] = df_metadata[['Is_AT1', 'Is_EM', 'Is_Non_IG', 'Is_Hybrid']].astype(float)
        
        # Set US T-Bills flags to 0 if it exists in the data (critical check)
        if 'US T-Bills' in df_metadata.index:
            df_metadata.loc['US T-Bills', ['Is_AT1', 'Is_EM', 'Is_Non_IG', 'Is_Hybrid']] = 0
        
        # Validate that all required columns exist and are numeric
        required_numeric_cols = ['Rating_Num', 'Duration', 'Current Yield Hdgd']
        for col in required_numeric_cols:
            if col not in df_metadata.columns:
                raise ValueError(f"Required column '{col}' not found in metadata")
            if not pd.api.types.is_numeric_dtype(df_metadata[col]):
                df_metadata[col] = pd.to_numeric(df_metadata[col], errors='coerce').fillna(0)
        
        return df_pct_change, df_metadata, df_common
    
    def optimise_portfolio(fund_name, returns, metadata, constraints, rf, target_return):
        """Portfolio optimization function"""
        mu = returns.mean().values * 252
        cov = LedoitWolf().fit(returns).covariance_ * 252
        
        idx = returns.columns.tolist()
        n = len(idx)
        w = cp.Variable(n)
        
        metadata = metadata.loc[idx]
        
        # Extract numeric columns - exactly as original file
        rating = pd.to_numeric(metadata['Rating_Num'], errors='coerce').fillna(0).values
        duration = pd.to_numeric(metadata['Duration'], errors='coerce').fillna(0).values
        yields = pd.to_numeric(metadata['Current Yield Hdgd'], errors='coerce').fillna(0).values / 100
        
        # Extract binary flags - exactly as original file
        is_at1 = pd.to_numeric(metadata['Is_AT1'], errors='coerce').fillna(0).values
        is_em = pd.to_numeric(metadata['Is_EM'], errors='coerce').fillna(0).values
        is_non_ig = pd.to_numeric(metadata['Is_Non_IG'], errors='coerce').fillna(0).values
        is_hybrid = pd.to_numeric(metadata['Is_Hybrid'], errors='coerce').fillna(0).values
        
        # Create constraints
        constraints_list = [cp.sum(w) == 1, w >= 0]
        constraints_list.append(is_non_ig @ w <= constraints['max_non_ig'])
        constraints_list.append(is_em @ w <= constraints['max_em'])
        constraints_list.append(is_at1 @ w <= constraints['max_at1'])
        constraints_list.append(is_hybrid @ w <= constraints['max_hybrid'])
        constraints_list.append(rating @ w >= constraints['min_rating'])
        
        # Initialize tbill_index variable
        tbill_index = None
        
        if constraints.get('max_tbill') is not None and 'US T-Bills' in idx:
            tbill_index = idx.index('US T-Bills')
            constraints_list.append(w[tbill_index] <= constraints['max_tbill'])
        
        if constraints['max_duration'] is not None:
            constraints_list.append(duration @ w <= constraints['max_duration'])
        
        constraints_list.append(mu @ w >= target_return)
        
        problem = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), constraints_list)
        
        try:
            problem.solve()
        except Exception as e:
            raise ValueError(f"Optimization solver error: {str(e)}")
        
        if w.value is None:
            raise ValueError("No solution found - constraints may be infeasible")
        
        weights = pd.Series(w.value, index=idx).round(4)
        weights = weights[weights > 0.001].sort_values(ascending=False)
        
        rating_avg = (rating @ w.value).item()
        tbill_weight = w.value[tbill_index].item() if tbill_index is not None else 0.0
        
        # Calculate metrics
        expected_return = (mu @ w.value).item()
        expected_volatility = np.sqrt((w.value).T @ cov @ w.value).item()
        avg_yield = (yields @ w.value).item()
        avg_duration = (duration @ w.value).item()
        em_exposure = (is_em @ w.value).item()
        at1_exposure = (is_at1 @ w.value).item()
        non_ig_exposure = (is_non_ig @ w.value).item()
        hybrid_exposure = (is_hybrid @ w.value).item()
        
        metrics = {
            'Expected Return': expected_return,
            'Expected Volatility': expected_volatility,
            'Avg Yield': avg_yield,
            'Avg Duration': avg_duration,
            'Avg Rating': rating_avg,
            'EM Exposure': em_exposure,
            'AT1 Exposure': at1_exposure,
            'Non-IG Exposure': non_ig_exposure,
            'Hybrid Exposure': hybrid_exposure,
            'T-Bill Exposure': tbill_weight
        }
        
        return weights, metrics
    
    def find_max_return_only(fund_name, returns, metadata, constraints):
        """Find maximum achievable return without risk minimization"""
        mu = returns.mean().values * 252
        
        idx = returns.columns.tolist()
        n = len(idx)
        w = cp.Variable(n)
        
        metadata = metadata.loc[idx]
        
        # Extract numeric columns
        rating = pd.to_numeric(metadata['Rating_Num'], errors='coerce').fillna(0).values
        duration = pd.to_numeric(metadata['Duration'], errors='coerce').fillna(0).values
        
        # Extract binary flags
        is_at1 = pd.to_numeric(metadata['Is_AT1'], errors='coerce').fillna(0).values
        is_em = pd.to_numeric(metadata['Is_EM'], errors='coerce').fillna(0).values
        is_non_ig = pd.to_numeric(metadata['Is_Non_IG'], errors='coerce').fillna(0).values
        is_hybrid = pd.to_numeric(metadata['Is_Hybrid'], errors='coerce').fillna(0).values
        
        # Create constraints (same as optimise_portfolio but without risk minimization)
        constraints_list = [cp.sum(w) == 1, w >= 0]
        constraints_list.append(is_non_ig @ w <= constraints['max_non_ig'])
        constraints_list.append(is_em @ w <= constraints['max_em'])
        constraints_list.append(is_at1 @ w <= constraints['max_at1'])
        constraints_list.append(is_hybrid @ w <= constraints['max_hybrid'])
        constraints_list.append(rating @ w >= constraints['min_rating'])
        
        # Initialize tbill_index variable
        tbill_index = None
        
        if constraints.get('max_tbill') is not None and 'US T-Bills' in idx:
            tbill_index = idx.index('US T-Bills')
            constraints_list.append(w[tbill_index] <= constraints['max_tbill'])
        
        if constraints['max_duration'] is not None:
            constraints_list.append(duration @ w <= constraints['max_duration'])
        
        # Maximize return instead of minimizing risk
        problem = cp.Problem(cp.Maximize(mu @ w), constraints_list)
        
        try:
            problem.solve()
        except Exception as e:
            raise ValueError(f"Max return optimization error: {str(e)}")
        
        if w.value is None:
            raise ValueError("No solution found for max return optimization")
        
        max_return = (mu @ w.value).item()
        return max_return
    
    def generate_efficient_frontier(fund_name, df_returns, df_metadata, fund_constraints, rf_rate_hist, step_size=0.0015):
        """Generate efficient frontier"""
        # Find minimum return portfolio
        try:
            w_min, m_min = optimise_portfolio(
                fund_name, df_returns, df_metadata, fund_constraints, rf_rate_hist,
                target_return = df_returns.mean().min() * 252 * 0.5
            )
        except Exception as e:
            raise ValueError(f"Cannot build min return portfolio for {fund_name}: {e}")
        
        min_return = m_min['Expected Return']
        try:
            max_return_discovered = find_max_return_only(fund_name, df_returns, df_metadata, fund_constraints)
            max_return = max(max_return_discovered, min_return * 2)
        except:
            max_return = min_return * 2
        
        targets = np.arange(min_return, max_return + step_size, step_size)
        returns_list, risks_list, metrics_dict, weights_dict = [], [], {}, {}
        
        for i, target in enumerate(targets):
            try:
                w, m = optimise_portfolio(
                    fund_name, df_returns, df_metadata, fund_constraints, rf_rate_hist, target
                )
                returns_list.append(m['Expected Return'])
                risks_list.append(m['Expected Volatility'])
                label = f"Portfolio {len(metrics_dict)+1}"
                metrics_dict[label] = m
                weights_dict[label] = w
            except Exception as e:
                continue
        
        if not metrics_dict:
            raise ValueError("No feasible portfolios found for the efficient frontier.")
        
        df_metrics = pd.DataFrame(metrics_dict)
        df_weights = pd.concat(weights_dict.values(), axis=1)
        df_weights.columns = metrics_dict.keys()
        df_weights = df_weights.fillna(0).round(4)
        
        sharpe_hist = ((df_metrics.loc['Expected Return'] - rf_rate_hist) / df_metrics.loc['Expected Volatility']).round(2)
        df_metrics.loc['Sharpe (Hist Avg)'] = sharpe_hist
        
        return returns_list, risks_list, df_metrics, df_weights
    
    # Process data
    df_pct_change, df_metadata, df_common = process_data(df_raw, df_metadata_raw)
    
    # Calculate risk-free rate
    if 'US T-Bills' in df_pct_change.columns:
        rf_rate_hist = df_pct_change['US T-Bills'].mean() * 252
    else:
        rf_rate_hist = 0.0  # Default or user-defined fallback
    
    # Calculate annualized returns and volatility
    annualized_returns = df_pct_change.mean() * 252 * 100
    annualized_std = df_pct_change.std() * np.sqrt(252) * 100
    results_df = pd.DataFrame({
        'Annualised Return (%)': annualized_returns, 
        'Standard Deviation (%)': annualized_std
    }).sort_values(by='Annualised Return (%)', ascending=False)
    
    # Main analysis section
    st.header("Portfolio Construction Model")
    
    # Data overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Assets", len(df_pct_change.columns))
    with col2:
        st.metric("Time Period", f"{df_pct_change.index.min().date()} to {df_pct_change.index.max().date()}")
    with col3:
        st.metric("Trading Days", len(df_pct_change))
    
    # Run optimization for all funds
    st.header("Multi-Fund Portfolio Optimization")
    
    # Store results for all funds
    all_fund_results = {}
    
    # Process each fund
    for fund in fund_list:
        st.subheader(f"{fund} Optimization")
        
        # Display current constraints being used for optimization
        constraints = st.session_state.fund_constraints[fund]
        st.info(f"**Current constraints for {fund}:** AT1: {constraints['max_at1']*100:.1f}%, "
                f"EM: {constraints['max_em']*100:.1f}%, "
                f"Non-IG: {constraints['max_non_ig']*100:.1f}%, "
                f"Hybrid: {constraints['max_hybrid']*100:.1f}%, "
                f"Duration: {constraints['max_duration'] if constraints['max_duration'] else 'None'} yrs, "
                f"Min Rating: {inverse_rating_scale[constraints['min_rating']]}")
        
        try:
            # Show optimization status
            with st.spinner(f"Running {fund} portfolio optimization..."):
                returns_list, risks_list, df_metrics, df_weights = generate_efficient_frontier(
                    fund, df_pct_change, df_metadata, 
                    constraints, rf_rate_hist
                )
            st.success(f"✅ {fund} optimization completed successfully!")
            
            # Store results
            all_fund_results[fund] = {
                'returns_list': returns_list,
                'risks_list': risks_list,
                'df_metrics': df_metrics,
                'df_weights': df_weights
            }
            
        except Exception as e:
            st.error(f"Error in {fund} optimization: {str(e)}")
            continue
    
    # Comparative Analysis Section
    if all_fund_results:
        st.header("Fund Comparison Analysis")
        
        # Comparative Efficient Frontiers
        st.subheader("Comparative Efficient Frontiers")
        
        # Create combined efficient frontier plot
        fig_combined_frontier = go.Figure()
        fund_colors = {
            'GFI': RUBRICS_COLORS['blue'],
            'GCF': RUBRICS_COLORS['orange'], 
            'EYF': RUBRICS_COLORS['medium_blue']
        }
        
        optimal_portfolios = {}
        
        for fund, results in all_fund_results.items():
            returns_list = results['returns_list']
            risks_list = results['risks_list']
            df_metrics = results['df_metrics']
            
            # Find optimal portfolio for this fund
            if 'Sharpe (Hist Avg)' in df_metrics.index:
                sharpe_row = pd.to_numeric(df_metrics.loc['Sharpe (Hist Avg)'], errors='coerce').fillna(0)
                expected_return_row = pd.to_numeric(df_metrics.loc['Expected Return'], errors='coerce').fillna(0)
                
                max_sharpe = sharpe_row.max()
                max_sharpe_portfolios = sharpe_row[sharpe_row == max_sharpe].index.tolist()
                
                if len(max_sharpe_portfolios) == 1:
                    optimal_portfolio = max_sharpe_portfolios[0]
                else:
                    tie_break_returns = expected_return_row[max_sharpe_portfolios]
                    optimal_portfolio = tie_break_returns.idxmax()
                
                optimal_idx = df_metrics.columns.get_loc(optimal_portfolio)
                optimal_portfolios[fund] = {
                    'name': optimal_portfolio,
                    'return': returns_list[optimal_idx],
                    'risk': risks_list[optimal_idx],
                    'sharpe': sharpe_row[optimal_portfolio]
                }
            
            # Add frontier line
            fig_combined_frontier.add_trace(go.Scatter(
                x=risks_list,
                y=returns_list,
                mode='lines+markers',
                name=f'{fund} Efficient Frontier',
                line=dict(color=fund_colors[fund], width=2),
                marker=dict(color=fund_colors[fund], size=4)
            ))
            
            # Add optimal portfolio marker
            if fund in optimal_portfolios:
                opt = optimal_portfolios[fund]
                fig_combined_frontier.add_trace(go.Scatter(
                    x=[opt['risk']],
                    y=[opt['return']],
                    mode='markers',
                    name=f'{fund} Optimal',
                    marker=dict(
                        color=fund_colors[fund],
                        size=12,
                        symbol='star',
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate=f'<b>{fund} Optimal</b><br>Return: {opt["return"]*100:.2f}%<br>Risk: {opt["risk"]*100:.2f}%<br>Sharpe: {opt["sharpe"]:.4f}<extra></extra>'
                ))
        
        fig_combined_frontier.update_layout(
            title="Comparative Efficient Frontiers - All Funds",
            xaxis_title="Volatility (Standard Deviation)",
            yaxis_title="Expected Return",
            template="plotly_white",
            height=500,
            font=dict(family="Ringside", size=12),
            title_font=dict(family="Ringside", size=16),
            legend_font=dict(family="Ringside", size=11)
        )
        
        st.plotly_chart(fig_combined_frontier, use_container_width=True)
        
        # Comparative Sharpe Ratios
        st.subheader("Optimal Portfolio Comparison")
        
        if optimal_portfolios:
            # Create comparison table
            comparison_data = []
            for fund, opt in optimal_portfolios.items():
                comparison_data.append({
                    'Fund': fund,
                    'Expected Return': f"{opt['return']*100:.2f}%",
                    'Expected Volatility': f"{opt['risk']*100:.2f}%", 
                    'Sharpe Ratio': f"{opt['sharpe']:.4f}",
                    'Portfolio': opt['name']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Best performing fund
            best_fund = max(optimal_portfolios.items(), key=lambda x: x[1]['sharpe'])
            st.success(f"🏆 Best performing fund: **{best_fund[0]}** with Sharpe ratio of {best_fund[1]['sharpe']:.4f}")

else:
    # Welcome message when no file is uploaded
    st.markdown("""
    <div class="welcome-message" style="text-align: center; padding: 2rem;">
        <h2>Welcome to Portfolio Construction Model</h2>
        <p>Upload your Excel file to begin portfolio optimization analysis.</p>
        <p>The file should contain two sheets:</p>
        <ul style="text-align: left; display: inline-block;">
            <li><strong>Index List:</strong> Price histories for assets</li>
            <li><strong>Sheet2:</strong> Asset metadata (ratings, duration, etc.)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)