`layers.py`: Removed bias from `Linear` used in keys, queries and values
Fixed `mean` in `torch.nn.init.normal_` for biases

`process_data.py`: Added `__name__ == "__main__"`

`train.py`: Added `AdamW` with learning rate from `train_config`