# Implementing Transformers with PyTorch - GPT (Decoder)
> Language modelling with the WikiHow dataset



## Setup

1. Clone the repository and create a new Python virtual environment,

```commandline
$> git clone --depth=1 shubham0204/language-modelling-with-pytorch
$> cd language-modelling-with-pytorch
$> python -m venv project_env
$> ./project_env/bin/activate
```

After activating the virtual environment, install Python dependencies, 

```commandline
$> (project_env) pip install -r requirements.txt
```

2. The dataset is included in the `dataset/` directory as a text file. The next step is to execute the `process_data.py` script
which will read the articles from the text file and transform it into a tokenized corpus of sentences. 
The `process_data.py` requires an input directory (where the text file is) and an output directory to store 
the tokenized corpus along with `index2word` and `word2index` mappings, serialized as Python `pickle`.
The input and the output directories are specified with `data_path` and `data_tensors_path` respectively in 
project's configuration file `project_config.toml`. 

`data_path` defaults to the `dataset/` directory in the project's root.

```commandline
$> (project_env) python process_data.py
```

3. After execution of the `process_data.py` script, the `vocab_size` parameter gets changed in the `project_config.toml`.
We're now ready to train the model. The parameters within the `train` group in `project_config.toml` are used to
control the training process. See [Understanding `project_config.toml`](#understanding-projectconfigtoml) to get more 
details about the `train` parameters.

## Inference and pretrained model



## Understanding `project_config.toml`

The `project_config.toml` file contains all settings required by the project which are used by nearly all scripts in the 
project. Having a global configuration in a TOML file enhances control over the project and provides a *sweet spot* 
where all settings could be viewed/modified at once.

> While developing the project, I wrote a blog - [Managing Deep Learning Models Easily With TOML Configurations](https://towardsdatascience.com/managing-deep-learning-models-easily-with-toml-configurations-fb680b9deabe). Do check it out.

The following sections describe each setting that can be changed through `project_config.toml`

### `train` configuration

- `num_train_iter` (`int`): The number of iterations to be performed on the training dataset. Note, an iteration refers to the forward-pass
of a single batch of data, and a back-pass to update the parameters.
- `num_test_iter` (`int`): The number of iterations to be performed on the test dataset.
- `test_interval` (`int`): The number of iterations after which testing should be performed.
- `batch_size` (`int`): Number of samples present in a batch.
- `learning_rate` (`float`): The learning rate used by the `optimizer` in `train.py`
- `checkpoint_path` (`str`): Path where checkpoints will be saved during training.
- `wandb_logging_enabled` (`bool`): Enable/disable logging to Weights&Biases console in `train.py` 
- `wandb_project_name` (`str`): If logging is enabled, the name of the `project` that is to be used for W&B
- `resume_training` (`bool`):
- `resume_training_checkpoint_path` (`str`):
- `compile_model` (`bool`):

### `data` configuration

- `vocab_size` (`int`): The number of tokens in the vocabulary. This variable is set when `process_data.py` script is executed.
- `test_split` (`float`): 
- `seq_length` (`int`):
- `data_path` (`str`):
- `data_tensors_path` (`str`):

### `model` configuration

- `embedding_dim` (`int`):
- `num_blocks` (`int`):
- `num_heads_in_block` (`int`):
- `dropout` (`float`):

### `deploy` configuration

- `host` (`str`):
- `port` (`int`):

## Deployment

### API Endpoints with FastAPI

### Web app with Streamlit

