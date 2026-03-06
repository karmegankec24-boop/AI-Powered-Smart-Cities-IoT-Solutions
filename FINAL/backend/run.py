"""backend/run.py — Start the FastAPI server"""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

subprocess.run([
    sys.executable, "-m", "uvicorn",
    "server:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--reload",
], check=True)
