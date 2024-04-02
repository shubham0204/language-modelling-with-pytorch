import streamlit as st

from process_data import get_tokens
from utils import Predictor

st.title("GPT - WikiHow Dataset")

st.markdown("## Model")
model_path_box = st.text_input("Enter model path (to a .pt file) ...")
data_tensors_box = st.text_input("Enter data tensors directory path ...")
device_choice_box = st.radio("Inference Device: ", ("cpu", "cuda"))
num_tokens_box = st.number_input(
    "Number of tokens to generate: ", min_value=10, max_value=100, value=50, step=10
)

st.markdown("## Inference")
input_prompt = st.text_area("Enter initial prompt (64 words long) ...")
temperature = st.slider(
    "Temperature", min_value=1.0, max_value=5.0, value=1.0, step=1.0
)

if st.button("Predict"):
    predictor = Predictor(model_path_box, data_tensors_box, device_choice_box)
    output = predictor.predict_tokens(
        get_tokens(input_prompt), num_tokens_box, temperature
    )
    output = Predictor.beautify_output(output)
    st.write(output)
