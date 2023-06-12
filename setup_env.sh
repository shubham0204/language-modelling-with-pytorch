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
unzip -q -j poemsdataset.zip "topics/beach/*" -d "dataset/"
unzip -q -j poemsdataset.zip "topics/animal/*" -d "dataset/"
unzip -q -j poemsdataset.zip "topics/water/*" -d "dataset/"
unzip -q -j poemsdataset.zip "topics/weather/*" -d "dataset/"
unzip -q -j poemsdataset.zip "topics/winter/*" -d "dataset/"
unzip -q -j poemsdataset.zip "topics/star/*" -d "dataset/"
unzip -q -j poemsdataset.zip "topics/sea/*" -d "dataset/"
unzip -q -j poemsdataset.zip "topics/river/*" -d "dataset/"
unzip -q -j poemsdataset.zip "topics/rain/*" -d "dataset/"
unzip -q -j poemsdataset.zip "topics/nature/*" -d "dataset/"
unzip -q -j poemsdataset.zip "topics/green/*" -d "dataset/"

# Move scripts to current directory
mv Poem_Maker_Transformer/* .

# Download dependencies
pip3 install munch wandb toml fire

# Clean other directories/files
rm poemsdataset.zip
rm -r Poem_Maker_Transformer