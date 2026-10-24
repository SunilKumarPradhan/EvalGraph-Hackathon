from pathlib import Path

import pytest

from evalgraph.understanding.pipeline import understand

ZOO = Path(__file__).parent / "zoo"
ARCHER = Path(__file__).resolve().parents[3].parent / "cuda-archer"


def run(path: Path):
    return understand({"local_path": str(path)}, lambda *a, **k: None)


@pytest.fixture(scope="session")
def zoo():
    return {p.name: run(p) for p in sorted(ZOO.iterdir()) if p.is_dir()}


@pytest.fixture(scope="session")
def archer():
    return run(ARCHER)
