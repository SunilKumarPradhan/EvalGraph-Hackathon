import atexit
import os
import shutil
import tempfile

# Tests never call DIAL, and write to a throwaway data dir.
os.environ["EVALGRAPH_MOCK_LLM"] = "1"
os.environ["ARCHER_MOCK_LLM"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["EVALGRAPH_DATA_DIR"] = tempfile.mkdtemp(prefix="evalgraph-test-")
atexit.register(shutil.rmtree, os.environ["EVALGRAPH_DATA_DIR"], True)
