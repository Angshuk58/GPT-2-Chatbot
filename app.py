import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load tokenizer & model
model_path = "./gpt2_chatbot"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Streamlit UI
st.title("🤖 Chatbot using GPT-2")

user_input = st.text_input("You:", "")

if st.button("Generate Response"):
    if user_input:
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        output = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        st.text_area("Chatbot:", value=response, height=150)
    else:
        st.warning("Please enter a message!")

