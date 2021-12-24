#!/bin/bash
conda env export > environment.yml
pip freeze > requirements.txt

# remove torch_rw 
sed -i '/torch-rw/d' environment.yml