import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re

# --- Text Preprocessing Function (defined once) ---
def preprocess_text(text):
    text = text.lower() # Lowercasing
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

# Use st.cache_resource to load and process data only once
@st.cache_resource
def load_data_and_vectorize():
    try:
        df = pd.read_csv('Conversation.csv')
        # Drop the 'Unnamed: 0' column if it exists and is just an index
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        st.error("Conversation.csv not found. Please make sure it's in the same directory.")
        st.stop()

    # Apply preprocessing to the 'question' column
    df['processed_question'] = df['question'].apply(preprocess_text)

    # Initialize TF-IDF Vectorizer
    # max_features can be adjusted based on your dataset size and memory
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    # Fit and transform the processed questions
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_question'])

    return df, tfidf_vectorizer, tfidf_matrix

# Load cached data and vectorizer
df, tfidf_vectorizer, tfidf_matrix = load_data_and_vectorize()

# --- Chatbot Logic ---
def get_bot_response(user_query):
    # Preprocess the user query
    processed_user_query = preprocess_text(user_query)

    if not processed_user_query:
        return "Please type a question."

    # Transform the user query using the *same* fitted TF-IDF vectorizer
    user_tfidf = tfidf_vectorizer.transform([processed_user_query])

    # Calculate cosine similarity between user query and all processed questions
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Get the index of the most similar question
    most_similar_index = cosine_similarities.argmax()

    # Get the similarity score
    similarity_score = cosine_similarities[most_similar_index]

    # Set a similarity threshold. If the score is too low, the bot doesn't know the answer.
    # You might need to tune this threshold based on your dataset and desired strictness.
    if similarity_score < 0.3:  # Adjust this threshold as needed
        return "I'm sorry, I don't have a direct answer to that. Could you please rephrase?"
    else:
        # Retrieve the answer for the most similar question
        return df['answer'].iloc[most_similar_index]

# --- Streamlit UI ---
st.set_page_config(page_title="Customer Support Chatbot", layout="centered")

st.title("🗣️ Customer Support Chatbot")
st.markdown("""
Welcome! I'm a simple chatbot designed to respond to your queries based on our conversation dataset.
Type your question below and press Enter.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me a question..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = get_bot_response(prompt)
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.info("Note: This is a basic chatbot. For more advanced features like intent recognition or contextual conversations, consider using frameworks like Rasa or Dialogflow.")
