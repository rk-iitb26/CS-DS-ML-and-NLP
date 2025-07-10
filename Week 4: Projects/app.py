import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# hf deployed model
@st.cache_resource
def load_custom_model():
    model_path = "roveenk/gpt2assignment"     
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_custom_model()

# Generate next words using your model
def generate_next_word(prompt, max_length):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=inputs.shape[1] + max_length, do_sample=True, top_k=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# streamlit
st.title("Custom GPT-2 Next Word Prediction")
st.write("Enter a prompt and see how your fine-tuned model completes it.")

prompt = st.text_area("Input Prompt", height=100, placeholder="Enter your prompt...")
max_len = st.slider("Max Output Length", 5, 100, 30)

if st.button("Generate"):
    if prompt.strip():
        with st.spinner("Generating..."):
            result = generate_next_word(prompt, max_len)
        st.success("Done!")
        st.write("### Generated Text:")
        st.write(result)
    else:
        st.warning("Please enter a prompt.")
