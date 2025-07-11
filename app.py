import streamlit as st
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Configuration & Model Loading ---
@st.cache_resource
def load_model():
    """Loads the conversational model and tokenizer from Hugging Face."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load the model. Please check your internet connection. Error: {e}")
        return None, None

# --- Knowledge Base Loading ---
def load_knowledge_base(file_path):
    """Loads the knowledge base from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Knowledge base file not found at: {file_path}")
        return []
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from the knowledge base file: {file_path}")
        return []

# --- Chatbot Logic ---
def get_response(user_query, knowledge_base, tokenizer, model, chat_history_ids):
    """
    Gets a response from the knowledge base or the local Hugging Face model.
    """
    # Check for a direct match in the knowledge base
    for qa_pair in knowledge_base:
        if user_query.lower() == qa_pair["question"].lower():
            return qa_pair["answer"], chat_history_ids

    # If no match, use the local conversational model
    try:
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(user_query + tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if len(st.session_state.messages) > 1 else new_user_input_ids

        # Generate a response while limiting the total chat history to 1024 tokens
        chat_history_ids = model.generate(bot_input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id)

        # Decode the last response from the bot
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response, chat_history_ids

    except Exception as e:
        st.error(f"An error occurred with the local model: {e}")
        return "Sorry, I'm having a little trouble thinking right now.", chat_history_ids


# --- Streamlit App ---
st.set_page_config(page_title="Intelligent Chatbot", page_icon="🤖")

st.title("🤖 Intelligent Open-Source Chatbot")
st.caption("This chatbot uses a local knowledge base and a Hugging Face conversational model.")

# Load model and tokenizer
tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]
# Initialize chat history tensor
if "chat_history" not in st.session_state:
    st.session_state.chat_history = torch.tensor([])


# Load the knowledge base
knowledge_base = load_knowledge_base('data/knowledge_base.json')

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, chat_history_ids = get_response(prompt, knowledge_base, tokenizer, model, st.session_state.chat_history)
            st.session_state.chat_history = chat_history_ids
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
