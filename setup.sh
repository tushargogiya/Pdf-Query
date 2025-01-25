#!/bin/bash

# Pre-install torch and dependencies
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Install xformers explicitly
pip install xformers==0.0.20 --no-build-isolation
