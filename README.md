# Marketing Mix Modeling (MMM) Dashboard

## Project Overview
This project is a decision-support tool designed to quantify the incremental impact of multi-channel marketing spend on weekly sales. By analyzing approximately 120 weeks of pulsed media activity, this project identifies the memory (Adstock) and effectiveness of various channels to optimize future budget allocation.

## Project Goals
* **Attribution**: Accurately rank marketing channels (Google, Facebook, Affiliates, etc.) based on their contribution to revenue.
* **Simulation**: Provide an interactive "What-If" environment for stakeholders to test spend scenarios.
* **Transparency**: Use SHAP (SHapley Additive exPlanations) to demystify model predictions.
* **Optimization**: Identify long-term anchor channels vs. short-term tactical triggers.

## Methodology
* **Algorithm**: Random Forest Regressor (Champion Model $R^2 = 0.78$).
* **Feature Engineering**: 
    * **Adstock Transformation**: Capturing lagged effects using a decay formula $A_t = T_t + \lambda A_{t-1}$.
    * **Temporal Controls**: Isulating 15% organic seasonality (Month dummies) and long-term trends.
    * **Fixed Effects**: Controlling for regional heterogeneity across 26 organizational divisions.
* **Validation**: Chronological 80/20 train-test split to prevent look-ahead bias.

## Folder Structure

```text
MARKETING MIX MODELLING/
├── data/                   
│   ├── Processed/          # Cleaned, engineered, and adstocked data
│   └── Raw/                # Original weekly marketing records
├── models/                 
│   ├── feature_names.pkl   # List of features used during training
│   ├── MMM Random forest model.pkl  # Trained 0.78 R2 model
│   └── scaler.pkl          # Saved Z-score scaling parameters
├── notebooks/
│   └── notebook.ipynb      # EDA, model training, and alpha selection logic
├── reports/
│   ├── Executive_Summary   # Business-facing PDF/Docx reports
│   └── Technical_Report    # Methodology-heavy PDF/Docx reports
├── src/
│   └── app.py              # Streamlit dashboard source code
├── .gitignore              # Files to ignore in Git (venv, .pyc, etc.)
├── README.md               # Project documentation
└── requirements.txt        # List of Python dependencies
```


## Usage Instructions

To run this dashboard locally, follow these steps. Ensure you have **Python 3.9+** installed on your system.

## 1. Clone the Repository

If you haven't already, clone this project to your local machine:
```bash
git clone https://github.com/N-anle/Marketing-Mix-Modelling
cd "MARKETING MIX MODELLING"
```

## 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to avoid dependency conflicts:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Required Packages

Install all necessary libraries, including Streamlit, Scikit-learn, and SHAP, using the requirements file:
```bash
pip install -r requirements.txt
```

## 4. Run the Dashboard

Launch the Streamlit application. The dashboard will automatically open in your default web browser
```bash
streamlit run src/app.py
```

## 5. Troubleshooting

- **File Not Found:** Ensure you are running the command from the root `MARKETING MIX MODELLING` folder so the script can find the paths to `../models/` and `../data/`.
- **Missing Pickle Files:** Verify that `MMM Random forest model.pkl` and `feature_names.pkl` are present in the `models/` directory.
````