from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaEmbeddings
import streamlit as st
import json

JSON_FILE_PATH = "../json/data.json"

# This loads the file needed to be called
def load_json_data(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
    return json_data

def processed_json_data(json_data):
    documents = []
    for record in json_data['workouts']:
        for exercise in record['exercises']:
            sets_text = ", ".join([f"{set['reps']} reps at {set.get('weight', 'N/A')}" for set in exercise.get('sets', [])])
            text = f"{record['date']} {record['type']}: {exercise['name']} ({exercise['muscleGroup']}), {sets_text}"
            documents.append({"content": text})
    return documents

json_data = load_json_data(JSON_FILE_PATH)
documents = processed_json_data(json_data)

embedding_model = LlamaEmbeddings(model="llama3")
vector_store = FAISS.from_documents(documents, embedding_model)
retriever = vector_store.as_retriever()

model = OllamaLLM(model="llama3")

template = """
You are a Training Assistant for the Klein Strength Training App, here to provide expert support to app users. Your goal is to help users understand and navigate the app’s elite-level strength coaching, answer questions about their training program, and support them in achieving their fitness goals. When responding, incorporate the Klein Strength Training App’s commitment to personalized, adaptable, and effective strength training.

Key Philosophy to Include in Responses:

Personalized & Adaptive Training: Klein Strength Training App programs are tailored based on the user’s unique strength levels, goals, lifting experience, and even sport-specific needs.

Support for All Levels: While athletes aiming for specific sport goals are a focus, the app is designed for anyone looking to build strength, muscle, or lose weight, offering flexibility for training in different environments (home, garage gym, or commercial gym).

Experienced Coaching: Every training program is crafted by a dedicated team with over 25 years of combined experience in strength and conditioning, led by a PhD in the field, ensuring a safe and effective experience.

Empowering Users: The Klein Strength Training App is more than a tool—it’s a training partner, designed to support users on every step of their strength journey with a focus on achieving their goals.

USER DATA: {user_data}

Answer the following question as a motivational speaker focusing on helping users hit their exercise goals.
Question: {question}
"""

prompt_template = ChatPromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm = model,
    retriever = retriever,
    prompt_template = prompt_template,
    return_source_documents = True
)

st.set_page_config(page_title="Klein Strength Training App", layout="wide")
st.sidebar.header("Chatbot Settings")
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark", "Sporty"])
persona = st.sidebar.selectbox("Choose AI Persona", ["Motivational", "Professional", "Friendly"])

st.title("Enhanced AI Chatbot - Fitness Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    role, message = chat

    if role == "User":
        st.markdown(f"**You** {message}")

    else:
        st.markdown(f"**AI** {message}")

user_question = st.text_input("Ask a question")