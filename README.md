# Intermediary Asset Pricing Replication Project

This repository contains our final project for replicating tables from the paper "Intermediary Asset Pricing: New Evidence from Many Asset Classes." Our goal is to reproduce key tables (Table 2 and Table 3) from the paper using data from CRSP, Compustat, and Datastream—and to update these results with the latest available data.

---

## Project Overview

The paper argues that capital shocks to financial intermediaries can explain cross-sectional differences in expected returns across various asset classes (stocks, bonds, options, commodities, currencies, and credit default swaps). Based on this idea, our project focuses on:
- **Building Risk Factors:** We construct risk factors using financial intermediaries' capital ratios.
- **Replicating Table 2:** This table shows the relative size of major market makers by calculating monthly ratios of total assets, book debt, book equity, and market equity relative to different market groups, and then averaging these ratios over time.
- **Replicating Table 3:**
  - *Panel A:* Uses data from 1970Q1 to 2012Q4 to compute the Market Capital Ratio, Book Capital Ratio, and AEM Leverage Ratio, and explores the correlations among these ratios and key economic variables (such as E/P, unemployment, GDP, financial conditions, and market volatility).
  - *Panel B:* Constructs risk factors from the ratios in Panel A and analyzes their correlations with each other and with the growth rates of various economic indicators.

---

## Data & Methodology

- **Data Sources:**  
  We modified the primary dealer list based on real data sources. The `ticks.csv` file was updated according to the corresponding gvkey codes.
  
- **Modifications to Calculations:**  
  We adjusted the calculation logic for important ratios and macroeconomic variables (e.g., key ratios and updated macro variable computations) based on the paper’s description, resulting in significant optimization of the replicated results for Table 3.
  
- **Output Generation:**  
  The reproduced table results are automatically generated as LaTeX (.tex) files and saved in the output directory. Further data analysis, including descriptive statistics, correlation analysis, and visualization of ratio trends, was also performed.

- **Automation:**  
  We implemented complete automation of the project workflow, storing all results in the `_output` directory.

---

## Project Structure

- **LaTeX Documents:**  
  Auto-generated LaTeX files include the replicated tables and additional figures from the paper.
  
- **Jupyter Notebook:**  
  A comprehensive notebook demonstrates data processing, ratio calculations, table generation, and analysis results.

- **Python Code Files:**  
  Scripts for data download, processing, and metric calculations are included. Each file contains detailed docstrings describing its functionality.

- **Automation (dodo.py):**  
  The `dodo.py` file automates the entire project workflow including data fetching, processing, table generation, and even running the final notebook.

- **Test Files:**  
  Unit tests for both Table 2 and Table 3 have been developed to ensure the replication results closely match the original paper.

- **Additional Files:**  
  The repository also includes supplementary files such as a README, an environment example file (`.env.example`), and other auxiliary scripts to ensure a clear and rigorous project process.

---

## Work Division

- **Liu Junyuan:**  
  - Developed the code for auto-generating the LaTeX summary document.
  - Worked on replicating additional tables and figures from the paper.
  - Enhanced table analysis scripts and improved the README file.

- **Hanlu Ge:**  
  - Modified the primary dealer list and updated `ticks.csv` based on real data sources.
  - Developed the Jupyter Notebook showcasing data processing, ratio calculations, table generation, and analysis.
  - Implemented the automation script (`dodo.py`) and wrote tests to ensure accurate replication of the tables.

Regular and transparent communication was maintained throughout the project to ensure that both team members remained aligned and met project goals.

---

## Setup & Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Junyuanxx/Final-Project-32900.git
   cd Final-Project-32900

2. **Create and Activate the Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Configure .env File:**

  Ensure you have an .env file (or rename .env.example to .env).
  Inside .env, specify the paths for DATA_DIR and OUTPUT_DIR so that the scripts can locate input data and store all generated output files correctly.

5. **Run the Project:**
   ```bash
   set PYTHONWARNINGS=ignore::FutureWarning
   doit

## Contact

If you have any questions or suggestions, please feel free to reach out via GitHub issues or contact the project members directly.