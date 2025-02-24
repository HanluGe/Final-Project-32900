"""
dodo.py - 基于 doit 自动化运行项目模块

该文件依赖于项目中的 config.py（配置路径、日期、用户名等）以及 src 目录下的各个模块：
Table02Prep.py、Table03.py、Table03Load.py、Table03Analysis.py 等。
运行时，会依次完成 Table 2 和 Table 3 的数据拉取、处理和报表生成，
以及可选的 LaTeX 文档编译。
"""

import os
from pathlib import Path
import warnings
import subprocess

from src import config

warnings.filterwarnings("ignore")

def task_table02_main():
    """
    运行 Table02Prep.py，生成 Table 2 的 LaTeX 表格和图形。
    分别运行原始版本（UPDATED=False）和更新版本（UPDATED=True）。
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

def task_table03_main():
    """
    运行 Table03.py，生成 Table 3 的 LaTeX 表格、摘要统计和图形。
    分别调用原始版本和更新版本（UPDATED=True）。
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

def task_pull_fred_data():
    """
    调用 Table03Load 中的数据拉取函数，更新 FRED 历史数据与 Shiller PE 数据，
    数据将写入 config.DATA_DIR 指定的目录中。
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

def task_compile_latex_docs():
    """
    编译 LaTeX 汇总文档生成 PDF 文件。
    这里假设你在 config.OUTPUT_DIR 下有一个 combined_document.tex 文件，
    编译后的 PDF 文件将输出到 config.OUTPUT_DIR 中。
    """
    tex_directory = Path(config.OUTPUT_DIR)  # 使用配置中定义的 OUTPUT_DIR
    tex_file = "combined_document.tex"  # 请确保该文件存在于 OUTPUT_DIR 下
    target_pdf = tex_directory / "report_final.pdf"

    compile_command = ["latexmk", "-pdf", "-pdflatex=xelatex", tex_file]
    clean_command = ["latexmk", "-C"]

    def compile_tex():
        subprocess.run(compile_command, cwd=str(tex_directory), check=True)

    def clean_aux_files():
        subprocess.run(clean_command, cwd=str(tex_directory), check=True)

    return {
        "actions": [clean_aux_files, compile_tex],
        "targets": [str(target_pdf)],
        "file_dep": [str(tex_directory / tex_file)],
        "clean": True,
    }