"""Export helpers (firmware gain header)."""

from pathlib import Path

import numpy as np


def export_c_gain(K, path, name="Kinf"):
    """Write a 4x12 gain as a C array (same format as the old ``Kinf_*.txt``)."""
    K = np.asarray(K)
    rows = [", ".join(f"{v:.8e}f" for v in row) for row in K]
    body = ",\n".join(f"    {{{r}}}" for r in rows)
    text = f"static const float {name}[{K.shape[0]}][{K.shape[1]}] = {{\n{body}}};\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    return text
