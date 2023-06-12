`loss.py`: Added `perplexity` metric.

`train.py`: Prints `perplexity` instead of accuracy now. Added `LearningRateScheduler`

`process_data.py`: Added `[SEP]`, `[START]` and `[END]` tokens to sequences.

`setup_env.sh`: It now unzips poems from 10 natural-like topics.

`predict.py`: Added open-text generation with `num_tokens` argument.