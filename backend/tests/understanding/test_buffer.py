import json

from evalgraph.understanding.buffer import RepoBuffer, load_dir, notebook_to_code

from .conftest import ZOO


def test_load_skips_noise(tmp_path):
    (tmp_path / "node_modules" / "x").mkdir(parents=True)
    (tmp_path / "node_modules" / "x" / "index.js").write_text("bad", encoding="utf-8")
    (tmp_path / "app.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "bundle.min.js").write_text("x", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    (tmp_path / "config.yaml").write_text("top_k: 10", encoding="utf-8")
    b = load_dir(tmp_path, RepoBuffer(name="t"))
    assert set(b.files) == {"app.py", "config.yaml"}
    assert len(b.digest) == 16


def test_notebook_to_code_comments_magics():
    nb = {"cells": [{"cell_type": "markdown", "source": "# hi"},
                    {"cell_type": "code", "source": ["!pip install x\n", "%time\n", "x = 1"]}]}
    code = notebook_to_code(json.dumps(nb))
    assert "# !pip install x" in code and "# %time" in code and "x = 1" in code and "# hi" not in code


def test_zoo_notebook_buffered_as_code():
    b = load_dir(ZOO / "notebook_rag", RepoBuffer(name="nb"))
    assert "FAISS.from_texts" in b.files["rag_experiment.ipynb"]
