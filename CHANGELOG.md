`config.py`: Reads global configuration from 
`project_config.toml`

`process_data.py`: Removed command-line arguments 
and added config-based parameters

`train.py`: Removed command-line arguments 
and added config-based parameters

`predict.py`: Removed command-line arguments 
and added config-based parameters

`utils.py`: `input_seq` gets converted to a `torch.Tensor` 
and again to `list` for appending predicted tokens