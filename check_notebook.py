import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

notebook_path = "미래에셋자산운용/sample_project/jupyter/lec1_5_visualization_ForOnelineLecture.ipynb"
cwd = os.path.dirname(os.path.abspath(notebook_path))

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
# Use default kernel if python3 is not found or use specific one
ep = ExecutePreprocessor(timeout=600)

try:
    # Use the current python executable as the kernel
    ep.preprocess(nb, {'metadata': {'path': cwd}})
    print("Notebook executed successfully.")
except Exception as e:
    print(f"Error executing the notebook: {e}")
    # Find the cell that failed
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    print("\n--- Error in Cell ---")
                    print(cell.source)
                    print("--- Traceback ---")
                    print("\n".join(output.traceback))
