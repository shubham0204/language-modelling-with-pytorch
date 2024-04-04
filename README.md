# Implementing Transformers with PyTorch - GPT (Decoder)
> Language modelling with the WikiHow dataset by implementing a transformer model (decoder) in PyTorch. Text is read 
> from a static source (a text file), processed, tokenized for training. After training, the model can be inferred 
> from a CLI-tool, a web app and a HTTP API.

## Features

- Data, training and model configuration managed with a single TOML file
- HTTP API ([FastAPI](https://fastapi.tiangolo.com/)) and web-application ([Streamlit](https://streamlit.io/)) for testing the model
- Metrics logging with [Weights & Biases](https://wandb.ai/site)

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
- `learning_rate` (`float`): The learning rate used by the `optimizer` in `train.py`.
- `checkpoint_path` (`str`): Path where checkpoints will be saved during training.
- `wandb_logging_enabled` (`bool`): Enable/disable logging to Weights&Biases console in `train.py`.
- `wandb_project_name` (`str`): If logging is enabled, the name of the `project` that is to be used for W&B.
- `resume_training` (`bool`): If `True`, a checkpoint will be loaded and training will be resumed.
- `resume_training_checkpoint_path` (`str`): If `resume_training = True`, the checkpoint will be loaded from this path. 
- `compile_model` (`bool`): Whether to use `torch.compile` to speedup training in `train.py`

### `data` configuration

- `vocab_size` (`int`): Number of tokens in the vocabulary. This variable is set when `process_data.py` script is executed.
- `test_split` (`float`): Fraction of data used for testing the model.
- `seq_length` (`int`): Context length of the model. Input sequences of `seq_length` will be produced in `train.py` to train the model.
- `data_path` (`str`): Path of the text file containing the articles.
- `data_tensors_path` (`str`): Path of the directory where tensors of processed data will be stored.

### `model` configuration

- `embedding_dim` (`int`): Dimensions of the output embedding for `torch.nn.Embedding`.
- `num_blocks` (`int`): Number of blocks to be used in the transformer model. A single block contains `MultiHeadAttention`, 
`LayerNorm` and `Linear` layers. See `layers.py`.
- `num_heads_in_block` (`int`): Number of heads used in `MultiHeadAttention`.
- `dropout` (`float`): Dropout rate for the transformer.

### `deploy` configuration

- `host` (`str`): Host IP used to deploy API endpoints.
- `port` (`int`): Port through which API endpoints will be exposed.

## Deployment

The trained ML model can be deployed in two ways, 

* As a [Streamlit]() app
* As an API endpoint with [FastAPI]()

### Web app with Streamlit

The model can be used with a [StreamLit]() app easily,

```commandline
$> (project_env) streamlit run app.py
```

In the app, we need to select a model for inference, the `data_tensors_path` (required for tokenization) and other
 parameters like number of words to generate and the temperature.

### API Endpoints with FastAPI

The model can be accessed with REST APIs built with [FastAPI](),

```commandline
$> (project_env) uvicorn api:server
```

A `GET` request at the `/predict` endpoint with query parameters `prompt`, `num_tokens` and `temperature` can 
be initiated to generate a response from the model,

```commandline
curl --location 'http://127.0.0.1:8000/predict?prompt=prompt_text_here&temperature=1.0&num_tokens=100'
```


