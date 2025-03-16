r"""
LaTeXDocGenerator.py

Key points:
1) Only the 4th table in each group is shrunk (via adjustbox).
2) "Table 3 Replication" includes an extra paragraph of English text explaining the work done
   for the Table 3 replication.
3) Underscore escaping, skipping repeated lines about "There are significantly fewer..." etc.
4) No large chunk of code omitted; this is the final complete version.
"""

import os
import re
import subprocess
from pathlib import Path

##############################################################################
# Helper functions for underscore escaping and skipping repeated lines
##############################################################################

def escape_underscores_in_text(line: str) -> str:
    """
    Escapes underscores unless the line contains \label{ or \includegraphics,
    using a negative lookbehind '(?<!\\)_'.
    """
    if ("\\label{" in line) or ("\\includegraphics" in line):
        return line
    pattern = r'(?<!\\)_'
    return re.sub(pattern, r'\\_', line)

def load_tex_content_no_env(tex_file: Path) -> list[str]:
    """
    Reads lines from a .tex file, skipping:
      - \begin{table}, \end{table}, \caption{
      - repeated descriptive lines about "There are significantly fewer entries..." etc.
    Escapes underscores for other lines unless skipping them.
    """
    skip_phrases = [
        "There are significantly fewer entries for book equity",
        "There are also some negatives for book equity",
        "than for other measures as shown in the count rows."
    ]

    lines_out = []
    if not tex_file.exists():
        return [f"% WARNING: File not found: {tex_file.name}"]

    with open(tex_file, "r", encoding="utf-8") as f:
        for original_line in f:
            line = original_line.rstrip("\n")

            # skip environment lines
            if any(token in line for token in ["\\begin{table", "\\end{table}", "\\caption{"]):
                continue

            # skip repeated lines
            if any(phrase in line for phrase in skip_phrases):
                continue

            # underscore escaping if not skipping
            lines_out.append(escape_underscores_in_text(line))
    return lines_out

##############################################################################
# Building table env: first 3 => no shrink, 4th => shrink
##############################################################################

def build_table_env_no_shrink(tex_file: Path, custom_title: str, descriptive_text: str = "") -> list[str]:
    """
    \begin{table}[H], no auto-numbering, no adjustbox => no shrink.
    """
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{escape_underscores_in_text(custom_title)}}}")

    if descriptive_text.strip():
        lines.append(r"\begin{flushleft}")
        for desc_line in descriptive_text.splitlines():
            lines.append(escape_underscores_in_text(desc_line))
        lines.append(r"\end{flushleft}")

    body = load_tex_content_no_env(tex_file)
    lines.extend(body)

    lines.append(r"\end{table}")
    return lines

def build_table_env_shrink(tex_file: Path, custom_title: str, descriptive_text: str = "") -> list[str]:
    """
    \begin{table}[H], uses adjustbox => shrink if wide, no auto-numbering.
    """
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{escape_underscores_in_text(custom_title)}}}")

    if descriptive_text.strip():
        lines.append(r"\begin{flushleft}")
        for desc_line in descriptive_text.splitlines():
            lines.append(escape_underscores_in_text(desc_line))
        lines.append(r"\end{flushleft}")

    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    body = load_tex_content_no_env(tex_file)
    lines.extend(body)
    lines.append(r"\end{adjustbox}")

    lines.append(r"\end{table}")
    return lines

##############################################################################
# Figures
##############################################################################

def build_figure_env(image_filename: str, custom_title: str = "", descriptive_text: str = "") -> list[str]:
    """
    \begin{figure}[H], optional caption, optional flushleft text.
    """
    lines = []
    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")

    if custom_title.strip():
        lines.append(f"\\caption{{{escape_underscores_in_text(custom_title)}}}")

    if descriptive_text.strip():
        lines.append(r"\begin{flushleft}")
        for desc_line in descriptive_text.splitlines():
            lines.append(escape_underscores_in_text(desc_line))
        lines.append(r"\end{flushleft}")

    # skip underscore escaping for the \includegraphics line
    lines.append(f"\\includegraphics[width=\\linewidth]{{{image_filename}}}")
    lines.append(r"\end{figure}")
    return lines

##############################################################################
# Main logic
##############################################################################

def main():
    # Output directory
    output_dir = (Path(__file__).resolve().parent.parent / "_output").resolve()
    output_dir.mkdir(exist_ok=True)

    combined_tex_filename = "combined_document.tex"
    combined_tex_path = output_dir / combined_tex_filename

    latex_lines = []

    # Preamble
    latex_lines.append(r"\documentclass{article}")
    latex_lines.append(r"\usepackage[utf8]{inputenc}")
    latex_lines.append(r"\usepackage{graphicx}")
    latex_lines.append(r"\usepackage{geometry}")
    latex_lines.append(r"\usepackage{xcolor}")
    latex_lines.append(r"\usepackage{adjustbox}")
    latex_lines.append(r"\usepackage{booktabs}")
    latex_lines.append(r"\usepackage{amsmath}")
    latex_lines.append(r"\usepackage{amssymb}")
    latex_lines.append(r"\usepackage{caption}")
    latex_lines.append(r"\usepackage{float}")
    latex_lines.append(r"\captionsetup{labelformat=empty}")
    latex_lines.append(r"\geometry{left=1in, right=1in, top=1in, bottom=1in}")
    latex_lines.append(r"\begin{document}")

    # Title
    latex_lines.append(r"\title{Intermediary asset pricing: New evidence from many asset classes}")
    latex_lines.append(r"\author{Hanlu Ge and Junyuan Liu}")
    latex_lines.append(r"\date{}")
    latex_lines.append(r"\maketitle")

    # Introduction
    latex_lines.append(r"\section{Introduction}")

    intro_p1 = r"""In this Final Project, our main task is to reproduce Table 2 and Table 3 from the paper "Intermediary asset pricing: New evidence from many asset classes" and to carry out a series of extension works based on this. Our specific work is divided into the following parts:"""
    for line in intro_p1.splitlines():
        latex_lines.append(escape_underscores_in_text(line))

    bullet_points = [
        "Modify the primary dealer list (ticks.csv) based on real data sources.",
        "Adjust the calculation methods for key ratios and macroeconomic variables in Table 2 and Table 3 according to the description in the paper.",
        "Automatically generate and save the reproduced table results as .tex files, and further perform data analysis such as descriptive statistics, correlation analysis, and trend plots of factors.",
        "Write additional files and implement project automation, such as the notebook, dodo.py, README file, and test files."
    ]
    latex_lines.append(r"\begin{itemize}")
    for item in bullet_points:
        latex_lines.append(f"\\item {escape_underscores_in_text(item)}")
    latex_lines.append(r"\end{itemize}")

    intro_p2 = r"""Through the above work, we have successfully optimized the reproduction based on the reference code, making the reproduced results extremely close to the target results while achieving clear visualization and an automated project workflow."""
    for line in intro_p2.splitlines():
        latex_lines.append(escape_underscores_in_text(line))

    ########################################################
    # Table 2 Replication
    ########################################################
    latex_lines.append(r"\section{Table 2 Replication}")

    # sub-item 1) table02.tex => no shrink
    latex_lines.extend(build_table_env_no_shrink(
        output_dir / "table02.tex",
        "Table 2 Replication"
    ))
    # sub-item 2) table02_figure.png => figure
    latex_lines.extend(build_figure_env(
        "table02_figure.png",
        "",
        "In the graphs below, we can see each of the four ratios shown over the original timeframe of 1960 to 2012."
    ))
    # sub-item 3) table02_corr.tex => no shrink
    latex_lines.extend(build_table_env_no_shrink(
        output_dir / "table02_corr.tex",
        "Table 2 Correlation Analysis"
    ))
    # sub-item 4) table02_sstable => shrink
    latex_lines.extend(build_table_env_shrink(
        output_dir / "table02_sstable.tex",
        "Table 2 Descriptive Statistics",
        "There are significantly fewer entries for book equity than for other measures as shown in the count rows. There are also some negatives for book equity."
    ))

    ########################################################
    # Table 2 (Updated)
    ########################################################
    latex_lines.append(r"\section{Table 2 (Updated)}")
    latex_lines.append("Below is the Table 2 result calculated using updated data up to 2025-02-01.")

    # 1) updated_table02.tex => no shrink
    latex_lines.extend(build_table_env_no_shrink(
        output_dir / "updated_table02.tex",
        "Table 2(Updated)"
    ))
    # 2) updated_table02_figure.png => figure
    latex_lines.extend(build_figure_env(
        "updated_table02_figure.png",
        "",
        "In the graphs below, we can see each of the four ratios shown over the updated timeframe of 1960 to 2024."
    ))
    # 3) updated_table02_corr.tex => no shrink
    latex_lines.extend(build_table_env_no_shrink(
        output_dir / "updated_table02_corr.tex",
        "Table 2 Correlation Analysis(Updated)"
    ))
    # 4) updated_table02_sstable.tex => shrink
    latex_lines.extend(build_table_env_shrink(
        output_dir / "updated_table02_sstable.tex",
        "Table 2 Descriptive Statistics(Updated)",
        "There are significantly fewer entries for book equity than for other measures as shown in the count rows. There are also some negatives for book equity."
    ))

    ########################################################
    # Table 3 Replication
    ########################################################
    latex_lines.append(r"\section{Table 3 Replication}")

    table3_intro = r"""Next, we replicate Table 3. We made many key logic corrections, including important ratio calculation methods, macroeconomic data sources, and computational methods. As a result, we have greatly optimized the reproduction performance, with most correlations being very close to the original table's results."""
    for line in table3_intro.splitlines():
        latex_lines.append(escape_underscores_in_text(line))

    # 1) table03.tex => no shrink
    latex_lines.extend(build_table_env_no_shrink(
        output_dir / "table03.tex",
        "Table 3(Replication)"
    ))
    # 2) table03_figure.png => figure
    latex_lines.extend(build_figure_env(
        "table03_figure.png",
        "Three Key Ratios Chart",
        "The figure below shows the trends of three key ratios. They closely match the original paper, confirming the good replication results."
    ))
    # 3) table03_figure03.png => figure
    latex_lines.extend(build_figure_env(
        "table03_figure03.png",
        "Variable Trend Chart"
    ))
    # 4) table03_sstable => shrink
    latex_lines.extend(build_table_env_shrink(
        output_dir / "table03_sstable.tex",
        "Table 3 Descriptive Statistics"
    ))

    ########################################################
    # Table 3 (Updated)
    ########################################################
    latex_lines.append(r"\section{Table 3 (Updated)}")
    latex_lines.append("Below is the Table 3 result calculated using updated data up to 2025-02-01.")

    # 1) updated_table03.tex => no shrink
    latex_lines.extend(build_table_env_no_shrink(
        output_dir / "updated_table03.tex",
        "Table 3(Updated)"
    ))
    # 2) updated_table03_figure.png => figure
    latex_lines.extend(build_figure_env(
        "updated_table03_figure.png",
        "Three Key Ratios Chart (Updated)"
    ))
    # 3) updated_table03_figure03.png => figure
    latex_lines.extend(build_figure_env(
        "updated_table03_figure03.png",
        "Variable Trend Chart (Updated)"
    ))
    # 4) updated_table03_sstable.tex => shrink
    latex_lines.extend(build_table_env_shrink(
        output_dir / "updated_table03_sstable.tex",
        "Table 3 Descriptive Statistics(Updated)"
    ))

    latex_lines.append(r"\end{document}")

    final_tex = "\n".join(latex_lines)
    with open(combined_tex_path, "w", encoding="utf-8") as f:
        f.write(final_tex)

    print(f"Merged LaTeX file generated at: {combined_tex_path}")

    # xelatex
    try:
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", combined_tex_filename],
            cwd=output_dir,
            check=True,
            timeout=60
        )
        print("XeLaTeX compilation successful!")
    except subprocess.CalledProcessError as e:
        print("XeLaTeX compilation failed (xelatex returned an error):", e)
    except subprocess.TimeoutExpired:
        print("xelatex command timed out.")

if __name__ == "__main__":
    main()
