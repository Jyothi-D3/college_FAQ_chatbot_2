import ast, sys, warnings
warnings.filterwarnings("ignore")

with open("governance.py") as f:
    src = f.read()
try:
    ast.parse(src)
    print("syntax OK")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    sys.exit(1)

from governance import (
    init_gov_db, get_all_results, get_suite_results,
    output_safety_filter, generate_report, apply_fixes,
    ALL_SUITES, REPORT_OUT
)
print(f"imports OK — {len(ALL_SUITES)} suites: {list(ALL_SUITES.keys())}")

filtered = output_safety_filter("My key is sk-abcdef1234567890123456 and I use memory.db")
print(f"filter OK: {filtered}")
