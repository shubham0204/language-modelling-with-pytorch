`layers.py`: Added CUDA `device`

`process_data`: Removed `Fire` dependency. Added print statements 
for showing no. of samples in train/test datasets

`setup_env.sh`: Unzips `forms/sonnet` and `forms/epic` from the dataset 
to increase training samples

`train.py`: Added `tqdm` progressbar for training and testing loops