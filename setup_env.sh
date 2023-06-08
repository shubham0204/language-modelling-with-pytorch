#!/usr/bin/env bash

# Install Kaggle CLI
pip3 install -q kaggle

# Move Kaggle API credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
ls ~/.kaggle

# Download the dataset
kaggle datasets download -d michaelarman/poemsdataset

# Unzip the dataset and copy the text files
mkdir dataset
unzip -q -j poemsdataset.zip "forms/carol/*" -d "dataset/"
unzip -q -j poemsdataset.zip "forms/epic/*" -d "dataset/"
unzip -q -j poemsdataset.zip "forms/sonnet/*" -d "dataset/"

# Move scripts to current directory
mv Poem_Maker_Transformer/* .

# Download dependencies
pip3 install munch wandb toml fire

# Clean other directories/files
rm poemsdataset.zip
rm -r Poem_Maker_Transformer