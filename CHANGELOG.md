`train.py`: Added `torch.compile` and `torch.save` now saves `state_dict` of `model` and `optimizer`

`project_config.toml`: Added `compile_model`, `resume_training` and `resume_training_checkpoint_path` settings

`process_data.py`: Renamed `token_number` and `token_linebreak`