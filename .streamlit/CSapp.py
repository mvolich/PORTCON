import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import t
from statsmodels.stats.weightstats import DescrStatsW
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Spreads & 12 Month Returns Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Spreads & 12 Month Returns Analysis")
st.markdown("Upload your Excel file to analyze excess returns by spread categories and fixed income categories.")

# File upload
uploaded_file = st.file_uploader(
    "Choose an Excel file",
    type=['xlsx', 'xls'],
    help="Upload an Excel file with 'Excess Return' sheet containing spread and return data"
)

@st.cache_data
def load_and_process_data(file):
    """Load and process the uploaded Excel file"""
    try:
        spreadsheet = pd.ExcelFile(file)
        data = spreadsheet.parse(sheet_name='Excess Return')
        
        # Prepare and clean data for categorization
        categories = []
        for i in range(0, data.shape[1], 3):
            sub_df = data.iloc[:, i:i+3].dropna()
            sub_df.columns = ['Date', 'Spread', '1 Yr Ahead ER']
            category_name = data.columns[i+1].replace(' OAS', '').strip()
            sub_df['Category'] = category_name
            categories.append(sub_df)

        combined_df = pd.concat(categories, ignore_index=True)
        
        # Ensure Date column is datetime
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        
        return combined_df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def categorize_spread(level):
    """Categorize spread levels into bins"""
    level_bps = level * 100  # convert from % to bps
    if level_bps < 100:
        return '<100'
    elif 100 <= level_bps < 150:
        return '100-150'
    elif 150 <= level_bps < 200:
        return '150-200'
    elif 200 <= level_bps < 250:
        return '200-250'
    elif 250 <= level_bps < 300:
        return '250-300'
    elif 300 <= level_bps < 400:
        return '300-400'
    elif 400 <= level_bps < 600:
        return '400-600'
    elif 600 <= level_bps < 800:
        return '600-800'
    else:
        return '800+'

def create_confidence_interval_plot(combined_df):
    """Create the confidence interval plot"""
    # Apply categorization
    combined_df['Spread Category'] = combined_df['Spread'].apply(categorize_spread)
    
    # Define desired order
    category_order = ['<100', '100-150', '150-200', '200-250', '250-300', '300-400', '400-600', '600-800', '800+']
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for idx, cat in enumerate(combined_df['Category'].unique()):
        confidence_intervals = []
        for spread_cat in category_order:
            subset = combined_df[(combined_df['Category'] == cat) & (combined_df['Spread Category'] == spread_cat)]['1 Yr Ahead ER']
            if len(subset) > 1:
                stats = DescrStatsW(subset)
                ci_low, ci_high = stats.tconfint_mean(alpha=0.05)
                mean = stats.mean
                confidence_intervals.append({
                    'Spread Category': spread_cat,
                    'Mean': mean,
                    'CI Lower': ci_low,
                    'CI Upper': ci_high,
                })

        if confidence_intervals:
            confidence_df = pd.DataFrame(confidence_intervals)
            color = colors[idx % len(colors)]

            fig.add_trace(go.Scatter(
                x=confidence_df['Spread Category'],
                y=confidence_df['Mean'],
                mode='lines+markers',
                name=f'{cat} Mean',
                line=dict(color=color, width=2),
                marker=dict(size=8),
                hoverinfo='text',
                hovertext=[f"{cat} Mean ({scat}): {mean:.2f}%" for scat, mean in zip(confidence_df['Spread Category'], confidence_df['Mean'])],
                opacity=0.7,
                legendgroup=cat
            ))

            fig.add_trace(go.Scatter(
                x=confidence_df['Spread Category'],
                y=confidence_df['CI Upper'],
                mode='lines+markers',
                name=f'{cat} CI Upper',
                line=dict(color=color, width=1, dash='dot'),
                marker=dict(size=6),
                hoverinfo='text',
                hovertext=[f"{cat} CI Upper ({scat}): {ci_upper:.2f}%" for scat, ci_upper in zip(confidence_df['Spread Category'], confidence_df['CI Upper'])],
                opacity=0.7,
                legendgroup=cat,
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=confidence_df['Spread Category'],
                y=confidence_df['CI Lower'],
                mode='lines+markers',
                name=f'{cat} CI Lower',
                line=dict(color=color, width=1, dash='dot'),
                marker=dict(size=6),
                hoverinfo='text',
                hovertext=[f"{cat} CI Lower ({scat}): {ci_lower:.2f}%" for scat, ci_lower in zip(confidence_df['Spread Category'], confidence_df['CI Lower'])],
                opacity=0.7,
                legendgroup=cat,
                showlegend=False
            ))

    fig.update_layout(
        title='Interactive Plot: Mean and 95% Confidence Interval of Excess Return by Spread Category',
        xaxis_title='Spread Category',
        yaxis_title='Excess Return (%)',
        hovermode='closest',
        legend=dict(title='Fixed Income Categories', itemclick='toggle', itemdoubleclick='toggleothers'),
        height=600
    )
    
    return fig

def create_summary_table(combined_df):
    """Create summary statistics table"""
    combined_df['Spread Category'] = combined_df['Spread'].apply(categorize_spread)
    
    summary_list = []
    category_order = ['<100', '100-150', '150-200', '200-250', '250-300', '300-400', '400-600', '600-800', '800+']

    for cat in combined_df['Category'].unique():
        for spread_cat in category_order:
            subset = combined_df[
                (combined_df['Category'] == cat) &
                (combined_df['Spread Category'] == spread_cat)
            ]['1 Yr Ahead ER']

            if len(subset) > 1:
                stats = DescrStatsW(subset)
                mean = stats.mean
                ci_low, ci_high = stats.tconfint_mean(alpha=0.05)
                count = len(subset)

                summary_list.append({
                    'Category': cat,
                    'Spread Category': spread_cat,
                    'CI Lower': round(ci_low, 2),
                    'Mean': round(mean, 2),
                    'CI Upper': round(ci_high, 2),
                    'N': count
                })

    summary_df = pd.DataFrame(summary_list)
    
    # Sort to match desired format
    summary_df['Spread Category'] = pd.Categorical(
        summary_df['Spread Category'],
        categories=category_order,
        ordered=True
    )
    summary_df.sort_values(by=['Category', 'Spread Category'], inplace=True)
    
    return summary_df

def create_heatmap(summary_df):
    """Create heatmap visualization with positive=green, negative=red"""
    pivot_df = summary_df.pivot(index="Category", columns="Spread Category", values="Mean")
    
    # Define explicitly custom diverging colorscale (negative=red, positive=green)
    custom_colorscale = [
        [0.0, "darkred"],    # Strong negative
        [0.5, "white"],      # Neutral
        [1.0, "darkgreen"]   # Strong positive
    ]
    
    # Plot heatmap explicitly with custom colorscale
    fig = px.imshow(
        pivot_df,
        text_auto=True,
        aspect="auto",
        title="Heatmap of Mean Excess Return by Spread Category and Asset Class",
        labels=dict(x="Spread Category", y="Asset Class", color="Mean Excess Return (%)"),
        color_continuous_scale=custom_colorscale,
        color_continuous_midpoint=0
    )
    
    fig.update_layout(height=600)
    return fig, pivot_df

def create_violin_plot(combined_df):
    """Create violin plot with date range selector"""
    combined_df['Spread Category'] = combined_df['Spread'].apply(categorize_spread)
    
    spread_order = ['<100', '100-150', '150-200', '200-250',
                    '250-300', '300-400', '400-600', '600-800', '800+']

    # Date range selector
    min_date = combined_df['Date'].min()
    max_date = combined_df['Date'].max()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    
    # Category selector
    selected_category = st.selectbox("Select Category", ['All'] + list(combined_df['Category'].unique()))
    
    # Filter data
    filtered_df = combined_df[
        (combined_df['Date'] >= pd.Timestamp(start_date)) &
        (combined_df['Date'] <= pd.Timestamp(end_date))
    ]
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    # Create violin plot with custom colors
    fig = go.Figure()
    
    for spread_cat in spread_order:
        spread_data = filtered_df[filtered_df['Spread Category'] == spread_cat]['1 Yr Ahead ER']
        
        if not spread_data.empty:
            # Calculate mean to determine color (positive = green, negative = red)
            mean_value = spread_data.mean()
            color = '#2E8B57' if mean_value >= 0 else '#DC143C'  # Green for positive, Red for negative
            
            fig.add_trace(go.Violin(
                y=spread_data,
                x=[spread_cat] * len(spread_data),
                name=spread_cat,
                box_visible=True,
                meanline_visible=True,
                spanmode='hard',
                legendgroup=spread_cat,
                scalegroup=spread_cat,
                line=dict(color=color),
                fillcolor=color,
                opacity=0.7
            ))
    
    fig.update_layout(
        title=f"Excess Returns Distribution - {selected_category} ({start_date} to {end_date})",
        xaxis=dict(
            title="Spread Category",
            categoryorder="array",
            categoryarray=spread_order
        ),
        yaxis_title="Excess Return (%)",
        violingap=0.1,
        height=600
    )
    
    return fig

# Main app logic
if uploaded_file is not None:
    # Load and process data
    combined_df = load_and_process_data(uploaded_file)
    
    if combined_df is not None:
        st.success(f"✅ Data loaded successfully! Shape: {combined_df.shape}")
        
        # Display basic info
        st.subheader("📋 Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Observations", len(combined_df))
        with col2:
            st.metric("Categories", combined_df['Category'].nunique())
        with col3:
            st.metric("Date Range", f"{combined_df['Date'].min().strftime('%Y-%m-%d')} to {combined_df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Confidence Intervals", 
            "📊 Summary Table", 
            "🔥 Heatmap", 
            "🎻 Violin Plot",
            "📥 Download Data"
        ])
        
        with tab1:
            st.subheader("Confidence Interval Analysis")
            fig_ci = create_confidence_interval_plot(combined_df)
            st.plotly_chart(fig_ci, use_container_width=True)
        
        with tab2:
            st.subheader("Summary Statistics")
            summary_df = create_summary_table(combined_df)
            st.dataframe(summary_df, use_container_width=True)
            
            # Download button for summary
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary as CSV",
                data=csv,
                file_name="spread_summary.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.subheader("Heatmap Analysis")
            summary_df = create_summary_table(combined_df)
            fig_heatmap, pivot_df = create_heatmap(summary_df)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Download button for heatmap data
            csv_heatmap = pivot_df.to_csv()
            st.download_button(
                label="📥 Download Heatmap Data as CSV",
                data=csv_heatmap,
                file_name="heatmap_data.csv",
                mime="text/csv"
            )
        
        with tab4:
            st.subheader("Distribution Analysis")
            fig_violin = create_violin_plot(combined_df)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        with tab5:
            st.subheader("Download Processed Data")
            
            # Download original processed data
            csv_processed = combined_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data as CSV",
                data=csv_processed,
                file_name="processed_data.csv",
                mime="text/csv"
            )
            
            # Download as Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name='Processed Data', index=False)
                summary_df = create_summary_table(combined_df)
                summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
                pivot_df = create_summary_table(combined_df).pivot(index="Category", columns="Spread Category", values="Mean")
                pivot_df.to_excel(writer, sheet_name='Heatmap Data')
            
            output.seek(0)
            st.download_button(
                label="📥 Download All Data as Excel",
                data=output.getvalue(),
                file_name="spreads_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("👆 Please upload an Excel file to begin the analysis.")
    
    # Instructions
    st.markdown("""
    ### 📋 Instructions:
    1. **File Format**: Upload an Excel file (.xlsx or .xls)
    2. **Sheet Name**: The file should contain a sheet named 'Excess Return'
    3. **Data Structure**: Data should be organized in columns of 3 (Date, Spread, 1 Yr Ahead ER) for each category
    4. **Spread Format**: Spreads should be in decimal format (e.g., 0.015 for 150 bps)
    
    ### 📊 What you'll get:
    - **Confidence Interval Plot**: Interactive visualization of mean returns with 95% confidence intervals
    - **Summary Table**: Statistical summary by category and spread level
    - **Heatmap**: Visual representation of mean returns across categories
    - **Violin Plot**: Distribution analysis with date range selection
    - **Download Options**: Export results in CSV or Excel format
    """)

