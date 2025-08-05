import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import plotly.graph_objects as go
import plotly.express as px
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
    page_title="Rubrics MVO Portfolio Optimizer",
    page_icon="📊",
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
    .main-header {{
        background: linear-gradient(90deg, {RUBRICS_COLORS['blue']}, {RUBRICS_COLORS['medium_blue']});
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
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

# Header with logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if logo:
        st.image(logo, width=200)
    st.markdown('<div class="main-header"><h1>Rubrics MVO Portfolio Optimizer</h1></div>', unsafe_allow_html=True)

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
        
        # Fund selection
        selected_fund = st.selectbox(
            "Select Fund",
            ['GFI', 'GCF', 'EYF'],
            help="Choose the fund to analyze and modify constraints"
        )
        
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
        
        # Display current fund constraints
        st.subheader(f"Current Constraints for {selected_fund}")
        
        constraints = st.session_state.fund_constraints[selected_fund]
        
        # Non-editable constraints (display only)
        st.markdown("**Fixed Constraints (Cannot be changed):**")
        st.markdown(f"• Max Non-IG: {constraints['max_non_ig']*100:.1f}%")
        st.markdown(f"• Max EM: {constraints['max_em']*100:.1f}%")
        st.markdown(f"• Max T-Bills: {constraints['max_tbill']*100:.1f}%")
        
        # Editable constraints
        st.markdown("**Editable Constraints:**")
        
        # AT1 constraint
        if selected_fund == 'EYF':
            st.markdown("• Max AT1: 0.01% (Fixed for EYF)")
        else:
            new_max_at1 = st.slider(
                "Max AT1 Exposure",
                min_value=0.0,
                max_value=0.5,
                value=constraints['max_at1'],
                step=0.01,
                help="Maximum Additional Tier 1 capital exposure",
                key=f"at1_slider_{selected_fund}"
            )
            st.session_state.fund_constraints[selected_fund]['max_at1'] = float(new_max_at1)
        
        # Duration constraint
        if selected_fund == 'EYF':
            st.markdown("• Max Duration: None (Fixed for EYF)")
        else:
            new_max_duration = st.slider(
                "Max Duration (years)",
                min_value=1.0,
                max_value=10.0,
                value=constraints['max_duration'] if constraints['max_duration'] else 5.0,
                step=0.1,
                help="Maximum portfolio duration",
                key=f"duration_slider_{selected_fund}"
            )
            st.session_state.fund_constraints[selected_fund]['max_duration'] = float(new_max_duration)
        
        # Hybrid constraint
        new_max_hybrid = st.slider(
            "Max Hybrid Exposure",
            min_value=0.0,
            max_value=0.5,
            value=constraints['max_hybrid'],
            step=0.01,
            help="Maximum hybrid instrument exposure",
            key=f"hybrid_slider_{selected_fund}"
        )
        st.session_state.fund_constraints[selected_fund]['max_hybrid'] = float(new_max_hybrid)
        
        # Rating constraint
        rating_options = list(rating_scale.keys())
        current_rating = inverse_rating_scale[constraints['min_rating']]
        new_min_rating = st.selectbox(
            "Minimum Rating",
            options=rating_options,
            index=rating_options.index(current_rating),
            help="Minimum average portfolio rating",
            key=f"rating_select_{selected_fund}"
        )
        st.session_state.fund_constraints[selected_fund]['min_rating'] = rating_scale[new_min_rating]
        
        # Reset button
        if st.button("Reset to Defaults"):
            st.session_state.fund_constraints = default_constraints.copy()
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
    st.header("📊 Portfolio Analysis")
    
    # Data overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Assets", len(df_pct_change.columns))
    with col2:
        st.metric("Time Period", f"{df_pct_change.index.min().date()} to {df_pct_change.index.max().date()}")
    with col3:
        st.metric("Trading Days", len(df_pct_change))
    
    # Exploratory Data Analysis Section
    with st.expander("🔍 Exploratory Data Analysis", expanded=False):
        st.subheader("📊 Data Processing Workflow")
        
        # Show raw data structure
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Raw Data Structure (Index List):**")
            st.dataframe(df_raw.head(), use_container_width=True)
        
        with col2:
            st.write("**Metadata Structure (Sheet2):**")
            st.dataframe(df_metadata_raw.head(), use_container_width=True)
        
        # Show data processing steps
        st.write("**Data Processing Steps:**")
        st.write("1. **Date/Value Column Separation**: Identified date and value columns from raw data")
        st.write("2. **Common Date Alignment**: Found overlapping dates across all assets")
        st.write("3. **Data Merging**: Combined all series on common dates")
        st.write("4. **Return Calculation**: Computed percentage changes")
        st.write("5. **Metadata Alignment**: Matched asset names with metadata")
        
        # Show processed data summary
        st.write("**Processed Data Summary:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Assets", len(df_common.columns))
        with col2:
            st.metric("Time Period", f"{len(df_common)} days")
        with col3:
            st.metric("Date Range", f"{df_common.index.min().strftime('%Y-%m-%d')} to {df_common.index.max().strftime('%Y-%m-%d')}")
        
        # Show data coverage
        st.write("**Data Coverage Analysis:**")
        first_last_dates = {}
        for col in df_common.columns:
            first_date = df_common[col].first_valid_index()
            last_date = df_common[col].last_valid_index()
            first_last_dates[col] = {'First Date': first_date.strftime('%Y-%m-%d'), 'Last Date': last_date.strftime('%Y-%m-%d')}
        
        coverage_df = pd.DataFrame.from_dict(first_last_dates, orient='index')
        st.dataframe(coverage_df, use_container_width=True)
        
        st.subheader("📈 Time Series Analysis")
        
        # Percentage change over time
        fig_pct = go.Figure()
        for col in df_pct_change.columns:
            fig_pct.add_trace(go.Scatter(
                x=df_pct_change.index, 
                y=df_pct_change[col] * 100,  # Convert to percentage
                mode='lines', 
                name=col, 
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ))
        
        fig_pct.update_layout(
            title="Daily Returns Over Time",
            xaxis_title="Date",
            yaxis_title="Daily Return (%)",
            hovermode="closest",
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pct, use_container_width=True)
        
        # Rebased indices
        df_rebased = df_common / df_common.iloc[0] * 100
        fig_rebased = go.Figure()
        
        for col in df_rebased.columns:
            fig_rebased.add_trace(go.Scatter(
                x=df_rebased.index, 
                y=df_rebased[col],
                mode='lines', 
                name=col, 
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
        
        fig_rebased.update_layout(
            title="Rebased Index Values (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Index Value",
            hovermode="closest",
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_rebased, use_container_width=True)
        
        st.subheader("📊 Asset Performance Summary")
        
        # Enhanced performance summary with additional metrics
        performance_summary = pd.DataFrame({
            'Annualised Return (%)': annualized_returns,
            'Annualised Volatility (%)': annualized_std,
            'Sharpe Ratio': (annualized_returns / annualized_std).round(3),
            'Min Daily Return (%)': (df_pct_change.min() * 100).round(2),
            'Max Daily Return (%)': (df_pct_change.max() * 100).round(2)
        }).sort_values(by='Sharpe Ratio', ascending=False)
        
        st.dataframe(performance_summary, use_container_width=True)
        
        # Distribution analysis
        st.subheader("📊 Return Distributions")
        
        # Create interactive histogram with dropdown
        fig_dist = go.Figure()
        
        for i, col in enumerate(df_pct_change.columns):
            fig_dist.add_trace(go.Histogram(
                x=df_pct_change[col] * 100,  # Convert to percentage
                name=col,
                nbinsx=50,
                opacity=0.7,
                visible=False if i > 0 else True
            ))
        
        # Add buttons for each asset
        buttons = []
        for i, col in enumerate(df_pct_change.columns):
            visibility = [False] * len(df_pct_change.columns)
            visibility[i] = True
            buttons.append(
                dict(
                    label=col,
                    method="update",
                    args=[{"visible": visibility}]
                )
            )
        
        fig_dist.update_layout(
            title="Return Distribution by Asset (Use dropdown to switch)",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "top"
            }],
            height=500
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Summary statistics
        st.write("**Distribution Statistics:**")
        summary_stats = pd.DataFrame({
            'Skewness': df_pct_change.skew().round(3),
            'Kurtosis': df_pct_change.kurtosis().round(3),
            'VaR (95%)': (df_pct_change.quantile(0.05) * 100).round(2),
            'CVaR (95%)': (df_pct_change[df_pct_change <= df_pct_change.quantile(0.05)].mean() * 100).round(2)
        }).sort_values(by='Skewness', key=abs, ascending=False)
        st.dataframe(summary_stats, use_container_width=True)
        
        st.subheader("🔗 Asset Correlation Matrix")
        
        # Interactive correlation matrix
        correlation_matrix = df_pct_change.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig_corr.update_layout(
            title="Asset Correlation Matrix",
            xaxis_title="Assets",
            yaxis_title="Assets",
            height=600,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Additional insights
        st.subheader("💡 Key Insights")
        
        # Find highest and lowest correlations
        corr_matrix_no_diag = correlation_matrix.where(~np.eye(correlation_matrix.shape[0], dtype=bool))
        max_corr = corr_matrix_no_diag.max().max()
        min_corr = corr_matrix_no_diag.min().min()
        
        # Find the pair with highest correlation
        max_corr_pair = np.where(correlation_matrix == max_corr)
        min_corr_pair = np.where(correlation_matrix == min_corr)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Highest Correlation:**")
            st.write(f"{correlation_matrix.columns[max_corr_pair[1][0]]} vs {correlation_matrix.columns[max_corr_pair[0][0]]}: {max_corr:.3f}")
        
        with col2:
            st.write("**Lowest Correlation:**")
            st.write(f"{correlation_matrix.columns[min_corr_pair[1][0]]} vs {correlation_matrix.columns[min_corr_pair[0][0]]}: {min_corr:.3f}")
        
        # Best performing assets
        st.write("**Top 3 Assets by Sharpe Ratio:**")
        top_assets = performance_summary.head(3)
        for i, (asset, row) in enumerate(top_assets.iterrows(), 1):
            st.write(f"{i}. **{asset}**: Return: {row['Annualised Return (%)']:.2f}%, Vol: {row['Annualised Volatility (%)']:.2f}%, Sharpe: {row['Sharpe Ratio']:.3f}")
    
    # Run optimization for selected fund
    st.header(f"🎯 {selected_fund} Portfolio Optimization")
    
    # Display current constraints being used for optimization
    st.info(f"**Current constraints for {selected_fund}:** AT1: {st.session_state.fund_constraints[selected_fund]['max_at1']*100:.1f}%, "
            f"EM: {st.session_state.fund_constraints[selected_fund]['max_em']*100:.1f}%, "
            f"Non-IG: {st.session_state.fund_constraints[selected_fund]['max_non_ig']*100:.1f}%, "
            f"Hybrid: {st.session_state.fund_constraints[selected_fund]['max_hybrid']*100:.1f}%, "
            f"Duration: {st.session_state.fund_constraints[selected_fund]['max_duration'] if st.session_state.fund_constraints[selected_fund]['max_duration'] else 'None'} yrs, "
            f"Min Rating: {inverse_rating_scale[st.session_state.fund_constraints[selected_fund]['min_rating']]}")
    
    try:
        # Show optimization status
        with st.spinner("Running portfolio optimization with current constraints..."):
            returns_list, risks_list, df_metrics, df_weights = generate_efficient_frontier(
                selected_fund, df_pct_change, df_metadata, 
                st.session_state.fund_constraints[selected_fund], rf_rate_hist
            )
        st.success("✅ Optimization completed successfully!")
        
        # Efficient frontier plot
        st.subheader("Efficient Frontier")
        
        # Determine optimal portfolio for marking
        if 'Sharpe (Hist Avg)' in df_metrics.index:
            # Get Sharpe ratios and expected returns
            sharpe_row = pd.to_numeric(df_metrics.loc['Sharpe (Hist Avg)'], errors='coerce').fillna(0)
            expected_return_row = pd.to_numeric(df_metrics.loc['Expected Return'], errors='coerce').fillna(0)
            
            # Find the maximum Sharpe ratio
            max_sharpe = sharpe_row.max()
            
            # Find all portfolios with the maximum Sharpe ratio
            max_sharpe_portfolios = sharpe_row[sharpe_row == max_sharpe].index.tolist()
            
            if len(max_sharpe_portfolios) == 1:
                # Only one portfolio has the maximum Sharpe ratio
                optimal_portfolio = max_sharpe_portfolios[0]
            else:
                # Multiple portfolios have the same maximum Sharpe ratio
                # Select the one with the highest expected return
                tie_break_returns = expected_return_row[max_sharpe_portfolios]
                optimal_portfolio = tie_break_returns.idxmax()
            
            # Find the index of the optimal portfolio in the lists
            optimal_idx = df_metrics.columns.get_loc(optimal_portfolio)
            optimal_risk = risks_list[optimal_idx]
            optimal_return = returns_list[optimal_idx]
        
        # Create hover text
        hover_texts = []
        for i, portfolio in enumerate(df_metrics.columns):
            m = df_metrics[portfolio]
            hover_texts.append(
                f"<b>{portfolio}</b><br>"
                f"Expected Return: {m['Expected Return']*100:.2f}%<br>"
                f"Volatility: {m['Expected Volatility']*100:.2f}%<br>"
                f"Sharpe: {m['Sharpe (Hist Avg)']:.2f}<br>"
                f"Avg Yield: {m['Avg Yield']*100:.2f}%<br>"
                f"Avg Duration: {m['Avg Duration']:.2f} yrs"
            )
        
        fig_frontier = go.Figure()
        fig_frontier.add_trace(go.Scatter(
            x=risks_list,
            y=returns_list,
            mode='lines+markers',
            name='Efficient Frontier',
            text=hover_texts,
            hoverinfo='text',
            line=dict(color=RUBRICS_COLORS['blue'], width=3),
            marker=dict(color=RUBRICS_COLORS['orange'], size=8)
        ))
        
        # Add optimal portfolio marker
        if 'Sharpe (Hist Avg)' in df_metrics.index:
            fig_frontier.add_trace(go.Scatter(
                x=[optimal_risk],
                y=[optimal_return],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    color='yellow',
                    size=15,
                    symbol='star',
                    line=dict(color='white', width=2)
                ),
                text=[f"<b>Optimal Portfolio</b><br>Expected Return: {optimal_return*100:.2f}%<br>Volatility: {optimal_risk*100:.2f}%<br>Sharpe: {max_sharpe:.2f}"],
                hoverinfo='text'
            ))
        
        fig_frontier.update_layout(
            title=f"{selected_fund} Efficient Frontier",
            xaxis_title="Volatility (Standard Deviation)",
            yaxis_title="Expected Return",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_frontier, use_container_width=True)
        
        # Portfolio weights visualization
        st.subheader("Portfolio Composition Across Frontier")
        
        df_pct = df_weights * 100
        df_pct = df_pct.div(df_pct.sum(axis=0), axis=1) * 100
        
        fig_weights = go.Figure()
        
        for asset in df_pct.index:
            fig_weights.add_trace(go.Scatter(
                x=df_pct.columns,
                y=df_pct.loc[asset],
                mode='lines+markers',
                stackgroup='one',
                name=asset,
                hovertemplate="<b>Asset:</b> %{fullData.name}<br><b>Portfolio:</b> %{x}<br><b>Weight:</b> %{y:.2f}%"
            ))
        
        # Mark the optimal portfolio on the category axis
        if 'Sharpe (Hist Avg)' in df_metrics.index:
            fig_weights.add_shape(
                type="line",
                xref="x",  # treat x-values as category labels
                yref="paper",  # span full height
                x0=optimal_portfolio,
                x1=optimal_portfolio,
                y0=0,
                y1=1,
                line=dict(color="white", width=3, dash="dash")
            )
            # Add a label above it
            fig_weights.add_annotation(
                x=optimal_portfolio,
                y=1.02,
                xref="x",
                yref="paper",
                text="Optimal Portfolio",
                showarrow=False,
                yanchor="bottom",
                font=dict(color="white")
            )
        
        fig_weights.update_layout(
            title=f"{selected_fund} Portfolio Composition",
            xaxis_title="Portfolios Along Efficient Frontier",
            yaxis_title="Weight (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_weights, use_container_width=True)
        
        # Metrics table
        st.subheader("Portfolio Metrics")
        
        # Create a clean, reliable metrics display
        # Create a fresh DataFrame for display with proper structure
        df_metrics_display = pd.DataFrame(index=df_metrics.index, columns=df_metrics.columns)
        
        # Define which rows need percentage conversion
        percent_rows = ['Expected Return', 'Expected Volatility', 'EM Exposure', 'AT1 Exposure',
                       'Non-IG Exposure', 'Hybrid Exposure', 'T-Bill Exposure', 'Avg Yield']
        
        # Process each row individually with explicit type handling
        for row in df_metrics.index:
            try:
                # Get the row data and ensure it's numeric
                row_data = df_metrics.loc[row].astype(float)
                
                if row in percent_rows:
                    # Convert to percentage and handle small values
                    percentage_values = []
                    for val in row_data:
                        # Convert to percentage
                        pct_val = val * 100
                        # Round to 2 decimal places
                        pct_val = round(pct_val, 2)
                        # Handle very small values
                        if abs(pct_val) < 0.01:
                            pct_val = 0.0
                        percentage_values.append(pct_val)
                    
                    df_metrics_display.loc[row] = percentage_values
                    
                elif row == 'Avg Rating':
                    # Convert to rating scale
                    rating_values = []
                    for val in row_data:
                        try:
                            rating_val = inverse_rating_scale.get(int(round(val)), f"{val:.2f}")
                            rating_values.append(rating_val)
                        except:
                            rating_values.append(f"{val:.2f}")
                    
                    df_metrics_display.loc[row] = rating_values
                    
                elif row == 'Avg Duration':
                    # Round duration values to 2 decimal places
                    duration_values = []
                    for val in row_data:
                        duration_values.append(round(val, 2))
                    
                    df_metrics_display.loc[row] = duration_values
                    
                else:
                    # Keep other rows as they are
                    df_metrics_display.loc[row] = row_data.values
                    
            except Exception as e:
                # Set to zeros as fallback
                df_metrics_display.loc[row] = [0.0] * len(df_metrics.columns)
        
        st.dataframe(df_metrics_display, use_container_width=True)
        
        # Constraints Budget Usage
        st.subheader("🔒 Constraints Budget Usage")
        
        # Get current constraints from session state
        current_constraints = st.session_state.fund_constraints[selected_fund]
        
        # Calculate constraint usage for each portfolio
        constraint_usage = {}
        
        for portfolio in df_metrics.columns:
            usage = {}
            
            # Get portfolio weights
            weights = df_weights[portfolio]
            
            # Ensure weights are numeric
            weights = pd.to_numeric(weights, errors='coerce').fillna(0)
            
            # Calculate constraint usage using current constraints
            # Ensure all values are numeric before calculations
            at1_exposure = (weights * pd.to_numeric(df_metadata.loc[weights.index, 'Is_AT1'], errors='coerce').fillna(0)).sum()
            em_exposure = (weights * pd.to_numeric(df_metadata.loc[weights.index, 'Is_EM'], errors='coerce').fillna(0)).sum()
            non_ig_exposure = (weights * pd.to_numeric(df_metadata.loc[weights.index, 'Is_Non_IG'], errors='coerce').fillna(0)).sum()
            hybrid_exposure = (weights * pd.to_numeric(df_metadata.loc[weights.index, 'Is_Hybrid'], errors='coerce').fillna(0)).sum()
            
            usage[f'AT1 (≤{current_constraints["max_at1"]*100:.1f}%)'] = (at1_exposure / current_constraints['max_at1']) * 100
            usage[f'EM (≤{current_constraints["max_em"]*100:.1f}%)'] = (em_exposure / current_constraints['max_em']) * 100
            usage[f'Non-IG (≤{current_constraints["max_non_ig"]*100:.1f}%)'] = (non_ig_exposure / current_constraints['max_non_ig']) * 100
            usage[f'Hybrid (≤{current_constraints["max_hybrid"]*100:.1f}%)'] = (hybrid_exposure / current_constraints['max_hybrid']) * 100
            
            # Handle T-Bills separately
            if 'US T-Bills' in weights.index:
                tbill_weight = pd.to_numeric(weights['US T-Bills'], errors='coerce')
                if pd.isna(tbill_weight):
                    tbill_weight = 0.0
                usage[f'T-Bills (≤{current_constraints["max_tbill"]*100:.1f}%)'] = (tbill_weight / current_constraints['max_tbill']) * 100
            else:
                usage[f'T-Bills (≤{current_constraints["max_tbill"]*100:.1f}%)'] = 0.0
            
            # Duration usage (if constraint exists)
            if current_constraints['max_duration'] is not None:
                duration_values = pd.to_numeric(df_metadata.loc[weights.index, 'Duration'], errors='coerce').fillna(0)
                avg_duration = (weights * duration_values).sum()
                usage[f'Duration (≤{current_constraints["max_duration"]} yrs)'] = (avg_duration / current_constraints['max_duration']) * 100
            else:
                usage['Duration (≤∞ yrs)'] = 0.0
            
            # Rating usage (showing how close to minimum rating)
            rating_values = pd.to_numeric(df_metadata.loc[weights.index, 'Rating_Num'], errors='coerce').fillna(0)
            avg_rating = (weights * rating_values).sum()
            min_rating = current_constraints['min_rating']
            rating_usage = ((avg_rating - min_rating) / (20 - min_rating)) * 100  # Scale from min_rating to AAA (20)
            usage[f'Rating (≥{inverse_rating_scale[min_rating]})'] = max(0, rating_usage)
            
            constraint_usage[portfolio] = usage
        
        # Create DataFrame for constraint usage
        df_constraint_usage = pd.DataFrame(constraint_usage).round(2)
        
        # Apply color coding to the dataframe
        def color_constraint_usage(val):
            if pd.isna(val):
                return ''
            if val > 90:
                return 'background-color: #ffcccc'  # Red for >90%
            elif val > 70:
                return 'background-color: #ffebcc'  # Orange for 70-90%
            else:
                return 'background-color: #ccffcc'  # Green for <70%
        
        # Display constraint usage table with color coding
        st.dataframe(df_constraint_usage.style.applymap(color_constraint_usage), use_container_width=True)
        
        # Visual representation of constraint usage
        st.subheader("Constraint Usage Visualization")
        
        # Create a bar chart showing constraint usage for the optimal portfolio
        if 'Sharpe (Hist Avg)' in df_metrics.index:
            # Get Sharpe ratios and expected returns
            sharpe_row = pd.to_numeric(df_metrics.loc['Sharpe (Hist Avg)'], errors='coerce').fillna(0)
            expected_return_row = pd.to_numeric(df_metrics.loc['Expected Return'], errors='coerce').fillna(0)
            
            # Find the maximum Sharpe ratio
            max_sharpe = sharpe_row.max()
            
            # Find all portfolios with the maximum Sharpe ratio
            max_sharpe_portfolios = sharpe_row[sharpe_row == max_sharpe].index.tolist()
            
            if len(max_sharpe_portfolios) == 1:
                # Only one portfolio has the maximum Sharpe ratio
                optimal_portfolio = max_sharpe_portfolios[0]
            else:
                # Multiple portfolios have the same maximum Sharpe ratio
                # Select the one with the highest expected return
                tie_break_returns = expected_return_row[max_sharpe_portfolios]
                optimal_portfolio = tie_break_returns.idxmax()
            
            # Get constraint usage for optimal portfolio
            optimal_usage = constraint_usage[optimal_portfolio]
            
            # Create bar chart
            fig_constraints = go.Figure()
            
            # Use the new constraint names with limits
            constraints_list = list(optimal_usage.keys())
            usage_values = list(optimal_usage.values())
            
            # Color bars based on usage level
            colors = ['red' if val > 90 else 'orange' if val > 70 else 'green' for val in usage_values]
            
            fig_constraints.add_trace(go.Bar(
                x=constraints_list,
                y=usage_values,
                marker_color=colors,
                text=[f'{val:.1f}%' for val in usage_values],
                textposition='auto'
            ))
            
            fig_constraints.update_layout(
                title=f"Constraint Usage for {optimal_portfolio}",
                xaxis_title="Constraints",
                yaxis_title="Usage (%)",
                yaxis=dict(range=[0, 100]),
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_constraints, use_container_width=True)
        
        # Optimal portfolio
        st.subheader("Optimal Portfolio")
        
        if 'Sharpe (Hist Avg)' in df_metrics.index:
            # Get Sharpe ratios and expected returns
            sharpe_row = pd.to_numeric(df_metrics.loc['Sharpe (Hist Avg)'], errors='coerce').fillna(0)
            expected_return_row = pd.to_numeric(df_metrics.loc['Expected Return'], errors='coerce').fillna(0)
            
            # Find the maximum Sharpe ratio
            max_sharpe = sharpe_row.max()
            
            # Find all portfolios with the maximum Sharpe ratio
            max_sharpe_portfolios = sharpe_row[sharpe_row == max_sharpe].index.tolist()
            
            if len(max_sharpe_portfolios) == 1:
                # Only one portfolio has the maximum Sharpe ratio
                optimal_portfolio = max_sharpe_portfolios[0]
            else:
                # Multiple portfolios have the same maximum Sharpe ratio
                # Select the one with the highest expected return
                tie_break_returns = expected_return_row[max_sharpe_portfolios]
                optimal_portfolio = tie_break_returns.idxmax()
            
            optimal_sharpe = sharpe_row[optimal_portfolio]
        else:
            st.error("Sharpe (Hist Avg) row not found in metrics")
            st.stop()
            
        optimal_return = df_metrics.loc['Expected Return', optimal_portfolio]
        optimal_vol = df_metrics.loc['Expected Volatility', optimal_portfolio]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Optimal Portfolio", optimal_portfolio)
        with col2:
            st.metric("Expected Return", f"{optimal_return:.2%}")
        with col3:
            st.metric("Expected Volatility", f"{optimal_vol:.2%}")
        with col4:
            st.metric("Sharpe Ratio", f"{optimal_sharpe:.2f}")
        
        # Optimal portfolio weights
        st.subheader("Optimal Portfolio Weights")
        optimal_weights = df_weights[optimal_portfolio].sort_values(ascending=False)
        optimal_weights = optimal_weights[optimal_weights > 0.001]
        
        fig_optimal = px.bar(
            x=optimal_weights.index,
            y=optimal_weights.values * 100,
            title=f"{selected_fund} Optimal Portfolio Weights",
            labels={'x': 'Asset', 'y': 'Weight (%)'},
            color_discrete_sequence=[RUBRICS_COLORS['blue']]
        )
        fig_optimal.update_layout(height=400)
        st.plotly_chart(fig_optimal, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in optimization: {str(e)}")
        st.write(f"Error type: {type(e)}")
        st.write(f"Error details: {e}")
        st.info("Try adjusting the constraints in the sidebar to make the optimization feasible.")

else:
    # Welcome message when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>Welcome to Rubrics MVO Portfolio Optimizer</h2>
        <p>Upload your Excel file to begin portfolio optimization analysis.</p>
        <p>The file should contain two sheets:</p>
        <ul style="text-align: left; display: inline-block;">
            <li><strong>Index List:</strong> Price histories for assets</li>
            <li><strong>Sheet2:</strong> Asset metadata (ratings, duration, etc.)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True) 