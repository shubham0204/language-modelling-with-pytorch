`layers.py`: Replaced `torch.sqrt` with `math.sqrt`

`process_data.py`: Added filtering for numbers, sentence breaks and word-contractions.

`utils.py`: Added `seq_length` as argument to class `Predictor`

`setup_env.sh`: Added `contractions` package for installation.

`predict.py`: Added `config.data.seq_length` to avoid hard-coding sequence length.