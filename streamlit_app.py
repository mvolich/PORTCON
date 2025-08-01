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
        if 'fund_constraints' not in st.session_state:
            st.session_state.fund_constraints = default_constraints.copy()
        
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
                help="Maximum Additional Tier 1 capital exposure"
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
                help="Maximum portfolio duration"
            )
            st.session_state.fund_constraints[selected_fund]['max_duration'] = float(new_max_duration)
        
        # Hybrid constraint
        new_max_hybrid = st.slider(
            "Max Hybrid Exposure",
            min_value=0.0,
            max_value=0.5,
            value=constraints['max_hybrid'],
            step=0.01,
            help="Maximum hybrid instrument exposure"
        )
        st.session_state.fund_constraints[selected_fund]['max_hybrid'] = float(new_max_hybrid)
        
        # Rating constraint
        rating_options = list(rating_scale.keys())
        current_rating = inverse_rating_scale[constraints['min_rating']]
        new_min_rating = st.selectbox(
            "Minimum Rating",
            options=rating_options,
            index=rating_options.index(current_rating),
            help="Minimum average portfolio rating"
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
        
        return df_pct_change, df_metadata
    
    def optimise_portfolio(fund_name, returns, metadata, constraints, rf, target_return):
        """Portfolio optimization function"""
        mu = returns.mean().values * 252
        cov = LedoitWolf().fit(returns).covariance_ * 252
        
        idx = returns.columns.tolist()
        n = len(idx)
        w = cp.Variable(n)
        
        metadata = metadata.loc[idx]
        
        # Extract numeric columns - exactly as original file
        rating = metadata['Rating_Num'].values
        duration = metadata['Duration'].values
        yields = metadata['Current Yield Hdgd'].values / 100
        
        # Extract binary flags - exactly as original file
        is_at1 = metadata['Is_AT1'].values
        is_em = metadata['Is_EM'].values
        is_non_ig = metadata['Is_Non_IG'].values
        is_hybrid = metadata['Is_Hybrid'].values
        
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
    df_pct_change, df_metadata = process_data(df_raw, df_metadata_raw)
    
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
    
    # Asset performance table
    st.subheader("Asset Performance Summary")
    st.dataframe(results_df, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Asset Correlation Matrix")
    correlation_matrix = df_pct_change.corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        color_continuous_scale='RdBu',
        aspect='auto',
        title="Asset Correlation Matrix"
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Run optimization for selected fund
    st.header(f"🎯 {selected_fund} Portfolio Optimization")
    
    try:
        returns_list, risks_list, df_metrics, df_weights = generate_efficient_frontier(
            selected_fund, df_pct_change, df_metadata, 
            st.session_state.fund_constraints[selected_fund], rf_rate_hist
        )
        
        # Efficient frontier plot
        st.subheader("Efficient Frontier")
        
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
        
        # Format metrics for display with debugging
        st.write("DEBUG: Starting metrics formatting...")
        st.write(f"DEBUG: df_metrics shape: {df_metrics.shape}")
        st.write(f"DEBUG: df_metrics index: {df_metrics.index.tolist()}")
        st.write(f"DEBUG: df_metrics dtypes: {df_metrics.dtypes}")
        
        # CRITICAL FIX: Ensure all metrics are numeric before processing
        st.write("DEBUG: Converting all metrics to numeric first...")
        df_metrics_display = df_metrics.copy()
        
        # Force all rows to be numeric with detailed debugging
        for row in df_metrics_display.index:
            st.write(f"DEBUG: Converting row '{row}' to numeric...")
            st.write(f"DEBUG: Original dtype: {df_metrics_display.loc[row].dtype}")
            st.write(f"DEBUG: Original sample values: {df_metrics_display.loc[row].head().tolist()}")
            
            # More robust conversion
            try:
                converted_row = pd.to_numeric(df_metrics_display.loc[row], errors='coerce').fillna(0)
                df_metrics_display.loc[row] = converted_row
                st.write(f"DEBUG: ✓ Row '{row}' converted successfully to {df_metrics_display.loc[row].dtype}")
            except Exception as e:
                st.write(f"DEBUG: ✗ Error converting row '{row}': {e}")
                # Force conversion by converting to string first, then to numeric
                try:
                    string_row = df_metrics_display.loc[row].astype(str)
                    converted_row = pd.to_numeric(string_row, errors='coerce').fillna(0)
                    df_metrics_display.loc[row] = converted_row
                    st.write(f"DEBUG: ✓ Row '{row}' converted via string method to {df_metrics_display.loc[row].dtype}")
                except Exception as e2:
                    st.write(f"DEBUG: ✗ String conversion also failed for '{row}': {e2}")
                    # Last resort: set to zeros
                    df_metrics_display.loc[row] = 0.0
                    st.write(f"DEBUG: ⚠️ Row '{row}' set to zeros as fallback")
        
        st.write(f"DEBUG: After numeric conversion - dtypes: {df_metrics_display.dtypes}")
        
        # CRITICAL: Force the entire DataFrame to be numeric at once
        st.write("DEBUG: Force converting entire DataFrame to numeric...")
        df_metrics_display = df_metrics_display.astype(float)
        st.write(f"DEBUG: After force conversion - dtypes: {df_metrics_display.dtypes}")
        
        percent_cols = ['Expected Return', 'Expected Volatility', 'EM Exposure', 'AT1 Exposure',
                       'Non-IG Exposure', 'Hybrid Exposure', 'T-Bill Exposure', 'Avg Yield']
        
        st.write(f"DEBUG: Processing {len(df_metrics_display.index)} rows...")
        
        for i, row in enumerate(df_metrics_display.index):
            st.write(f"DEBUG: Processing row {i+1}/{len(df_metrics_display.index)}: {row}")
            st.write(f"DEBUG: Row dtype before processing: {df_metrics_display.loc[row].dtype}")
            st.write(f"DEBUG: Row sample values: {df_metrics_display.loc[row].head().tolist()}")
            
            try:
                if row in percent_cols:
                    st.write(f"DEBUG: Converting {row} to percentage...")
                    # Ensure row is numeric before processing
                    if df_metrics_display.loc[row].dtype == 'object':
                        st.write(f"DEBUG: Row '{row}' is still object, forcing conversion...")
                        df_metrics_display.loc[row] = pd.to_numeric(df_metrics_display.loc[row], errors='coerce').fillna(0)
                    
                    # Row is now numeric, multiply by 100
                    df_metrics_display.loc[row] = df_metrics_display.loc[row] * 100
                    df_metrics_display.loc[row] = df_metrics_display.loc[row].round(2)
                    st.write(f"DEBUG: ✓ {row} converted successfully")
                elif row == 'Avg Rating':
                    st.write(f"DEBUG: Converting {row} to rating scale...")
                    # Ensure row is numeric before processing
                    if df_metrics_display.loc[row].dtype == 'object':
                        st.write(f"DEBUG: Row '{row}' is still object, forcing conversion...")
                        df_metrics_display.loc[row] = pd.to_numeric(df_metrics_display.loc[row], errors='coerce').fillna(0)
                    
                    # Row is now numeric, apply rating conversion
                    df_metrics_display.loc[row] = df_metrics_display.loc[row].apply(
                        lambda x: inverse_rating_scale.get(int(round(x)), f"{x:.2f}")
                    )
                    st.write(f"DEBUG: ✓ {row} converted successfully")
                else:
                    st.write(f"DEBUG: Skipping {row} (no conversion needed)")
            except Exception as e:
                st.write(f"DEBUG: ✗ Error processing {row}: {e}")
                st.write(f"DEBUG: Error type: {type(e)}")
                raise
        
        st.dataframe(df_metrics_display, use_container_width=True)
        
        # Optimal portfolio
        st.subheader("Optimal Portfolio")
        
        if 'Sharpe (Hist Avg)' in df_metrics.index:
            sharpe_row = pd.to_numeric(df_metrics.loc['Sharpe (Hist Avg)'], errors='coerce').fillna(0)
            optimal_portfolio = sharpe_row.idxmax()
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