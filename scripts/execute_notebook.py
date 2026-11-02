"""
Execute the notebook and save it with all cell outputs, charts, and metrics.
"""
import nbformat
from nbclient import NotebookClient
import os

nb_path = os.path.join(os.path.dirname(__file__), "notebooks", "AgriPrice_Forecasting_Way2Agri.ipynb")

print(f"Reading notebook from: {nb_path}")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print("Starting execution of all cells...")
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook execution completed! Output saved to: {nb_path}")
