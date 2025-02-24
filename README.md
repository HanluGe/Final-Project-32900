# Intermediary Asset Pricing Replication Project

This repository contains our final project for replicating tables from the "Intermediary Asset Pricing" paper. Our goal is to reproduce key tables (Table 2 and Table 3) from the paper using data from CRSP, Compustat, and Datastream—and to update these results with the latest available data.

---

## Project Overview

The paper argues that capital shocks to financial intermediaries can explain the cross-sectional differences in expected returns across various asset classes (stocks, bonds, options, commodities, currencies, and credit default swaps). Based on this idea, our project focuses on:
- **Building risk factors** using financial intermediaries' capital ratios.
- **Replicating Table 2:** This table shows the relative size of major market makers by calculating monthly ratios of total assets, book debt, book equity, and market equity relative to different market groups (bd, bank, compust) and averaging over time.
- **Replicating Table 3:**
  - *Panel A:* Uses data from 1970Q1 to 2012Q4 to compute the Market Capital Ratio, Book Capital Ratio, and AEM Leverage Ratio, and explores the correlations among these ratios and key economic variables (such as E/P, unemployment, GDP, financial conditions, and market volatility).
  - *Panel B:* Constructs risk factors from the ratios in Panel A and analyzes their correlations with each other and with the growth rates of various economic indicators.

---

## Data & Methodology

- **Data Sources:**  
  We gather data from CRSP, Compustat, and Datastream. Some data are also taken directly from the original paper.
  
- **Key Definitions:**  
  - **Capital Market Ratio (ηₜ):**  
  $$ 
  ηₜ = \frac{\sum_i \text{Market Equity}_{i,t}}{\sum_i (\text{Market Equity}_{i,t} + \text{Book Debt}_{i,t})}
  $$
  - **Book Capital Ratio:** The ratio of book equity to total book capital (book equity plus book debt).
  - **AEM Leverage Ratio:** Defined as the inverse of the book leverage (book debt/book equity) for broker-dealers.
  - **Risk Factor Construction:**  
    For each ratio, we run an AR(1) regression and use the residuals (normalized by the lagged value) to form the risk factors.

---

## Project Structure

- **LaTeX Document:**  
  Auto-generated LaTeX project summary that includes replicated tables and additional figures from the paper.
  
- **GitHub Setup:**  
  - This README file outlining the project.
  - Environment configuration files (e.g., `env`, `requirements.txt`).
  - Standardized file naming across the repository.

- **Jupyter Notebook:**  
  Demonstrates data processing, ratio calculations, table generation, and analysis results.

- **Python Code Files:**  
  Scripts for data download, processing, and metric calculations. Each Python file includes a docstring at the top explaining its functionality.

- **dodo.py:**  
  Implements project automation for end-to-end execution.

- **Test Files:**  
  Contains tests to verify that our replicated table data match the original paper’s results.

---

## Work Division

- **Liu Junyuan:**  
  - Develops code for auto-generating the LaTeX summary document.
  - Works on replicating additional tables and figures from the paper.
  - Enhances table analysis scripts and improves the README file.

- **Hanlu:**  
  - Develops the Jupyter Notebook to showcase our data processing, ratio calculations, table generation, and analysis.
  - Implements the automation script (`dodo.py`) and writes tests to ensure accurate replication of the tables.

We maintain transparent and regular communication throughout the project to ensure that both team members stay aligned and meet the project goals.

---

## Setup & Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/intermediary_asset_pricing.git
   cd intermediary_asset_pricing

2. **Create and Activate the Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Run the Project:**
  - Execute the Jupyter Notebook to view data analysis and replication steps.
  - Run `dodo.py` for the full automation process.

## Contact

If you have any questions or suggestions, please feel free to reach out via GitHub issues or contact the project members directly.