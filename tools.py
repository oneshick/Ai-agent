import sys
import pandas as pd
import numpy as np
from io import StringIO

from security import validate_code


def execute_python(code: str, df: pd.DataFrame) -> str:
    try:
        validate_code(code)
    except ValueError as e:
        return f"[SECURITY] Код заблокирован: {e}"

    local_vars = {"df": df.copy(), "pd": pd, "np": np}
    output_lines = []
    old_stdout = sys.stdout
    try:
        sys.stdout = buffer = StringIO()
        exec(code, local_vars)
        sys.stdout = old_stdout
        printed = buffer.getvalue()
        if printed:
            output_lines.append(printed.strip())
        lines = [l.strip() for l in code.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
        if lines:
            try:
                val = eval(lines[-1], local_vars)
                if val is not None:
                    if isinstance(val, pd.DataFrame):
                        output_lines.append(val.to_string())
                    elif isinstance(val, pd.Series):
                        output_lines.append(val.to_string())
                    else:
                        output_lines.append(str(val))
            except Exception:
                pass
        return "\n".join(output_lines) if output_lines else "Код выполнен успешно (нет вывода)"
    except Exception as e:
        sys.stdout = old_stdout
        return f"Ошибка выполнения: {e}"
