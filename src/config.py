"""Load project configurations from .env files.
Provides easy access to paths and credentials used in the project.
Meant to be used as an imported module.

If `config.py` is run on its own, it will create the appropriate
directories.

For information about the rationale behind decouple and this module,
see https://pypi.org/project/python-decouple/
"""

from decouple import config
from pathlib import Path

# 假设你的 config.py 位于项目 src/ 目录下，
# 通过 parent.parent 使 BASE_DIR 指向更上层，如 your_project/
BASE_DIR = Path(__file__).resolve().parent.parent

# 这里从 .env 读取 WRDS_USERNAME 或使用默认空值
WRDS_USERNAME = config("WRDS_USERNAME", default="")

# 如果 .env 里没指定 DATA_DIR，默认就用 your_project/my_data
DATA_DIR = config('DATA_DIR', default=(BASE_DIR / 'my_data'), cast=Path)
# 如果 .env 里没指定 OUTPUT_DIR，默认就用 your_project/my_output
OUTPUT_DIR = config('OUTPUT_DIR', default=(BASE_DIR / 'my_output'), cast=Path)

# 若需要 START_DATE, END_DATE, UPDATED_END_DATE 也可以在 .env 中定义
START_DATE = config('START_DATE', default='1960-01-01')
END_DATE = config('END_DATE', default='2012-12-31')
UPDATED_END_DATE = config('UPDATED_END_DATE', default='2024-02-29')

if __name__ == "__main__":
    # 如果直接执行 python src/config.py，会自动创建下面这些目录
    
    (DATA_DIR / 'pulled').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'manual').mkdir(parents=True, exist_ok=True)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Directories ensured:")
    print(f"  - DATA_DIR: {DATA_DIR}")
    print(f"  - OUTPUT_DIR: {OUTPUT_DIR}")
    
    # 你也可以在这里 print 一下 WRDS_USERNAME 或其他变量
    print(f"WRDS_USERNAME: {WRDS_USERNAME}")