import json
import pickle
import random
import numpy as np
import nltk
import streamlit as st
from nltk.stem.lancaster import LancasterStemmer
from tensorflow import keras

nltk.download("punkt", quiet=True)
stemmer = LancasterStemmer()

# --- Load Data ---
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

with open("data.pickle", "rb") as f:
    words, labels = pickle.load(f)[:2]   # adjust to how you saved pickle

model = keras.models.load_model("model.keras")

# --- Helper functions ---
def bag_of_words(sentence, words):
    tokens = nltk.word_tokenize(sentence)
    tokens = [stemmer.stem(w.lower()) for w in tokens if w.isalpha()]
    bag = [1 if w in tokens else 0 for w in words]
    return np.array([bag], dtype=np.float32)

def get_response(msg):
    bow = bag_of_words(msg, words)
    results = model.predict(bow)
    results_index = np.argmax(results)
    tag = labels[results_index]

    # choose a random response for that intent
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I didn’t understand. Try again!"

# --- Streamlit UI ---
st.title("🤖 Chatbot")
user_input = st.text_input("You:", "")

if st.button("Send") and user_input.strip():
    reply = get_response(user_input)
    st.write("**Bot:**", reply)
