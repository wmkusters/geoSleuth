#!/bin/sh

echo "\n>>>> Starting up jupyter notebook server...\n"
jupyter trust Example_Pipeline_Notebook.ipynb
jupyter notebook Example_Pipeline_Notebook.ipynb --ip 0.0.0.0 --port 8888 --no-browser --allow-root
