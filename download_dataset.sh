#!/usr/bin/env bash

# Install Kaggle CLI
pip3 install -q kaggle kaggle-cli

# Move Kaggle API credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
ls ~/.kaggle

# Download the dataset
kaggle datasets download -d michaelarman/poemsdataset

# Unzip the dataset and copy the text files
unzip -q poemsdataset.zip
mkdir dataset
cp -r forms/carol/*.txt dataset/

# Clean other directories/files
rm poemsdataset.zip
rm -r topics
rm -r forms
