#!/usr/bin/env bash

# Move scripts to current directory
mv Poem_Maker_Transformer/* .

# Download dependencies
pip3 install munch wandb toml fire contractions

# Clean other directories/files
rm -r Poem_Maker_Transformer