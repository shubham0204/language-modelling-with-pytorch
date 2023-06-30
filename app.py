from layers import Transformer
from utils import Predictor
from utils import load_dict_from_pickle
from process_data import get_tokens
import streamlit as st
import os
import torch

st.title( "GPT - WikiHow Dataset" )

st.markdown( "## Model" )
model_path_box = st.text_input( "Enter model path (to a .pt file) ..." )
data_tensors_box = st.text_input( "Enter data tensors directory path ..." )
device_choice_box = st.radio( "Inference Device: " , ( "cpu" , "cuda" ) )

st.markdown( "## Inference" )
input_prompt = st.text_area( "Enter initial prompt (64 words long) ..." )
temperature = st.slider( "Temperature" , min_value=1.0 , max_value=5.0 , value=1.0 , step=1.0 )

if st.button( "Predict" ):
    compute_device = torch.device(device_choice_box)
    print("Using device {} for inference".format(compute_device))
    checkpoint = torch.load(model_path_box, map_location=compute_device)
    config = checkpoint["config"]
    data_config = config.data
    model_config = config.model
    model = Transformer(
        vocab_size=data_config.vocab_size,
        embedding_dim=model_config.embedding_dim,
        seq_length=data_config.seq_length,
        num_blocks=model_config.num_blocks,
        num_heads_in_block=model_config.num_heads_in_block,
        dropout=model_config.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(compute_device)
    model.eval()

    idx_to_word = load_dict_from_pickle(os.path.join(data_tensors_box, "idx_to_word.pkl"))
    word_to_idx = load_dict_from_pickle(os.path.join(data_tensors_box, "word_to_idx.pkl"))
    predictor = Predictor(model, idx_to_word, word_to_idx, compute_device, temperature, config.data.seq_length)

    output = predictor.predict_tokens( get_tokens( input_prompt ) , 50 )
    output = Predictor.beautify_output(output)
    st.write( output )

