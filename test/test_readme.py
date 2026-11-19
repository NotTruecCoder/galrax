import contextlib
import io
import pathlib
import re

import jax
import pytest


README = pathlib.Path(__file__).resolve().parent.parent / "README.md"
BLOCKS = [
    block
    for block in re.findall(r"```python\n(.*?)```", README.read_text("utf-8"), re.S)
    if "R.<" not in block  # Sage syntax
]


@pytest.mark.slow
def test_readme_examples_run():
    pytest.importorskip("flint")
    namespace = {}
    # Users run with JAX defaults, not the strict promotion of conftest.py.
    with (
        jax.numpy_dtype_promotion("standard"),
        jax.numpy_rank_promotion("allow"),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        for block in BLOCKS:
            exec(compile(block, str(README), "exec"), namespace)
