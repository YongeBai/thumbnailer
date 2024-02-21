#!/bin/bash

pip install -r requirements.txt
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd ..