import sys
from pathlib import Path

# Force matplotlib backend and initialize early to avoid circular import issues in Streamlit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.streamlit_app import run_app

run_app()
