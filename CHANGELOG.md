`loss.py`: `logits` and `targets` are now reshaped before computing log-softmax

`process_data.py`: `n_gram_sequences` are now shuffled before saving to file.

`utils.py`: Added `temperature` parameter in softmax computation 

`predict.py`: Added `temperature` parameter in softmax computation and `map_location` in `torch.load`