`process_data.py`: Removed `PorterStemmer` and added regex to remove hashtags.

`train.py`: `config` is now saved in each model with `torch.save`. Only the best model gets saved overtime.

`predict.py`: The model is now constructed with the `config` loaded from `checkpoint`

`utils.py`: Added `beautify_output` method to format model output.