"""
dodo.py - Automates project tasks using doit.

This file relies on config.py for paths and src/ modules such as:
Table02Prep.py, Table03.py, Table03Load.py, Table03Analysis.py, etc.
It runs Table 2 and Table 3 data fetch, processing, and testing.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from pathlib import Path
import warnings
import subprocess

from src import config
warnings.filterwarnings("ignore")

def task_table02_main():
    """
    Runs Table02Prep.py to generate Table 2's LaTeX and figures.
    Executes both original (UPDATED=False) and updated (UPDATED=True) versions.
    """
    original_dir = os.getcwd()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    def create_original_table02():
        os.chdir(os.path.join(current_dir, "src"))
        os.system('python -c "import sys; sys.path.insert(0, \'src\'); import Table02Prep; Table02Prep.main()"')
        os.chdir(original_dir)

    def create_updated_table02():
        os.chdir(os.path.join(current_dir, "src"))
        os.system('python -c "import sys; sys.path.insert(0, \'src\'); import Table02Prep; Table02Prep.main(UPDATED=True)"')
        os.chdir(original_dir)

    return {
        'actions': [create_original_table02, create_updated_table02],
        'verbosity': 2,
    }

def task_test_table02():
    """
    Runs the unit tests for Table 02, located in Table02_testing.py.
    """
    original_dir = os.getcwd()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    def run_table02_tests():
        os.chdir(os.path.join(current_dir, "src"))
        os.system('python -m unittest Table02_testing.py')
        os.chdir(original_dir)

    return {
        'actions': [run_table02_tests],
        'verbosity': 2,
    }

def task_table03_main():
    """
    Runs Table03.py to generate Table 3's LaTeX, summary stats, and figures.
    Calls both original (UPDATED=False) and updated (UPDATED=True) versions.
    """
    original_dir = os.getcwd()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    def create_original_table03():
        os.chdir(os.path.join(current_dir, "src"))
        os.system('python -c "import sys; sys.path.insert(0, \'src\'); import Table03; Table03.main()"')
        os.chdir(original_dir)

    def create_updated_table03():
        os.chdir(os.path.join(current_dir, "src"))
        os.system('python -c "import sys; sys.path.insert(0, \'src\'); import Table03; Table03.main(UPDATED=True)"')
        os.chdir(original_dir)

    return {
        'actions': [create_original_table03, create_updated_table03],
        'verbosity': 2,
    }

def task_test_table03():
    """
    Runs the unit tests for Table 03, located in Table03_testing.py.
    """
    original_dir = os.getcwd()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    def run_table03_tests():
        os.chdir(os.path.join(current_dir, "src"))
        os.system('python -m unittest Table03_testing.py')
        os.chdir(original_dir)

    return {
        'actions': [run_table03_tests],
        'verbosity': 2,
    }

def task_pull_fred_data():
    """
    Calls Table03Load functions to update FRED historical data and Shiller PE data,
    writing outputs to config.DATA_DIR.
    """
    original_dir = os.getcwd()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    def pull_fred_past():
        os.chdir(os.path.join(current_dir, "src"))
        os.system('python -c "import sys; sys.path.insert(0, \'src\'); import Table03Load; Table03Load.load_fred_past()"')
        os.chdir(original_dir)

    def pull_shiller_pe():
        os.chdir(os.path.join(current_dir, "src"))
        os.system('python -c "import sys; sys.path.insert(0, \'src\'); import Table03Load; Table03Load.load_shiller_pe()"')
        os.chdir(original_dir)

    return {
        'actions': [pull_fred_past, pull_shiller_pe],
        'verbosity': 2,
    }

def main():
    pass

if __name__ == "__main__":
    main()
