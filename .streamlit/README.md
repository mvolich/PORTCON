# Rubrics MVO Portfolio Optimizer

A Streamlit web application for Mean-Variance Optimization (MVO) of fixed income portfolios, specifically designed for Rubrics Asset Management's three funds: GFI, GCF, and EYF.

## Features

- **Interactive Portfolio Optimization**: Real-time MVO analysis with customizable constraints
- **Fund-Specific Constraints**: Pre-configured constraints for GFI, GCF, and EYF funds
- **Dynamic Constraint Adjustment**: Modify fund constraints through an intuitive interface
- **Rubrics Branding**: Company colors and logo integration
- **Comprehensive Visualizations**: Efficient frontier plots, correlation matrices, and portfolio composition charts
- **Excel File Upload**: Support for Excel files with price history and metadata

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Upload your Excel file with the following structure:
   - **Sheet 1 (Index List)**: Price histories for assets
   - **Sheet 2 (Sheet2)**: Asset metadata including ratings, duration, and classification flags

3. Select a fund (GFI, GCF, or EYF) from the sidebar

4. Adjust fund constraints as needed:
   - **Fixed Constraints** (cannot be changed):
     - Max Non-IG Exposure
     - Max EM Exposure  
     - Max T-Bills Exposure
   - **Editable Constraints**:
     - Max AT1 Exposure (except EYF)
     - Max Duration (except EYF)
     - Max Hybrid Exposure
     - Minimum Rating

5. View the optimization results including:
   - Efficient frontier plots
   - Portfolio composition analysis
   - Performance metrics
   - Optimal portfolio identification

## Fund Constraints

### GFI (Global Fixed Income UCITS Fund)
- Max Non-IG: 25%
- Max EM: 30%
- Max AT1: 15%
- Max Duration: 6.5 years
- Min Rating: BBB-
- Max Hybrid: 15%
- Max T-Bills: 20%

### GCF (Global Credit UCITS Fund)
- Max Non-IG: 10%
- Max EM: 35%
- Max AT1: 10%
- Max Duration: 3.5 years
- Min Rating: BBB
- Max Hybrid: 10%
- Max T-Bills: 20%

### EYF (Enhanced Yield UCITS Fund)
- Max Non-IG: 100%
- Max EM: 100%
- Max AT1: 0.01% (fixed)
- Max Duration: None
- Min Rating: BB
- Max Hybrid: 20%
- Max T-Bills: 20%

## Technical Details

- **Optimization Engine**: CVXPY for convex optimization
- **Risk Estimation**: Ledoit-Wolf shrinkage estimator for robust covariance estimation
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas for data manipulation and analysis

## File Structure

```
├── streamlit_app.py      # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── AssetAlloc.py        # Original analysis file (for reference)
```

## Rubrics Branding

The application uses the official Rubrics color palette:
- **Rubrics Blue**: #001E4F
- **Rubrics Medium Blue**: #2C5697
- **Rubrics Light Blue**: #7BA4DB
- **Rubrics Grey**: #D8D7DF
- **Rubrics Orange**: #CF4520

The logo is automatically loaded from the Rubrics website and displayed in the application header.

## Support

For technical support or questions about the application, please contact the development team. 