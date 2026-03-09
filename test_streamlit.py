import subprocess
import requests
import time

p = subprocess.Popen(["python3", "-m", "streamlit", "run", "app.py", "--server.headless", "true"])
time.sleep(5)
try:
    response = requests.get("http://localhost:8501")
    print(response.status_code)
except Exception as e:
    print(e)
p.terminate()

