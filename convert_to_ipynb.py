import os
import json

def convert_py_to_ipynb():
    py_files = [
        "phase1_ingestion.py", 
        "phase2_preprocessing.py", 
        "phase3_model.py", 
        "phase4_training.py", 
        "phase5_evaluation.py", 
        "inference.py"
    ]
    
    for py_file in py_files:
        filepath = os.path.join(r"d:\heart_disease", py_file)
        if not os.path.exists(filepath):
            print(f"Warning: {py_file} not found.")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            code_content = f.read()
            
        # Manually constructing the simple Notebook JSON structure
        # This removes the need for the user to install any 3rd party nbformat packages!
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# {py_file.replace('.py', '')}\n", "Execute the cell below to run the code and save outputs directly into this notebook."]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [line + '\n' for line in code_content.split('\n')]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.11.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        ipynb_file = filepath.replace('.py', '.ipynb')
        with open(ipynb_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
            
        print(f"Successfully generated: {os.path.basename(ipynb_file)}")

if __name__ == '__main__':
    print("=== Converting Python Scripts to Jupyter Notebooks ===")
    convert_py_to_ipynb()
    print("All done! You can now open these .ipynb files in VSCode or Jupyter Lab.")
