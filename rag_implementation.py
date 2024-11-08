import streamlit as st
import json
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# Path to the JSON data file
JSON_FILE_PATH = "json/data.json"


# Load and process JSON data
def load_json_data(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
    return json_data

# Process and returns JSON data
def process_json_data(json_data):
    documents = []
    for record in json_data['workouts']:
        for exercise in record['exercises']:
            sets_text = ", ".join(
                [f"{set['reps']} reps at {set.get('weight', 'N/A')}" for set in exercise.get('sets', [])])
            text = f"{record['date']} - {record['type']}: {exercise['name']} ({exercise['muscleGroup']}), {sets_text}"
            documents.append(Document(page_content=text))
    return documents


# Load JSON data and prepare documents for RAG
json_data = load_json_data(JSON_FILE_PATH)
documents = process_json_data(json_data)

# Initialize Ollama Embeddings
embedding_model = OllamaEmbeddings(model="llama3")

# Initialize Chroma vector store
vector_store = Chroma.from_documents(documents, embedding_model, persist_directory=None)
retriever = vector_store.as_retriever()

# Initialize the LLM with Ollama
model = OllamaLLM(model="llama3")

# Define the response template
template = """
You are a {persona} Training Assistant for the Klein Strength Training App, here to provide expert support to app users. Your goal is to help users understand and navigate the app’s elite-level strength coaching, answer questions about their training program, and support them in achieving their fitness goals. Respond in a style that aligns with the theme: {theme}.

USER DATA: {user_data}

Question: {question}

Answer:
"""

# Create the ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(template)

# Set up the LLM chain using the prompt template
llm_chain = LLMChain(llm=model, prompt=prompt_template)

# Streamlit UI setup
st.set_page_config(page_title="Enhanced AI Chatbot", layout="wide")

# Sidebar for customization
st.sidebar.header("Chatbot Settings")
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark", "Sporty"])
persona = st.sidebar.selectbox("Choose AI Persona", ["Motivational", "Professional", "Friendly"])

# Apply CSS based on the selected theme
if theme == "Dark":
    st.markdown(
        """
        <style>
            .reportview-container {
                background-color: #333333;
                color: white;
            }
            .sidebar .sidebar-content {
                background-color: #444444;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
elif theme == "Sporty":
    st.markdown(
        """
        <style>
            .reportview-container {
                background-color: #0E1F40;
                color: #F5A623;
            }
            .sidebar .sidebar-content {
                background-color: #0E1F40;
                color: #F5A623;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
else:  # Light theme (default)
    st.markdown(
        """
        <style>
            .reportview-container {
                background-color: #f0f2f6;
                color: black;
            }
            .sidebar .sidebar-content {
                background-color: #ffffff;
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Main Chat Interface
st.title("Enhanced AI Chatbot - Fitness Assistant")

# Display Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat History in a customized layout
for chat in st.session_state.chat_history:
    role, message = chat
    if role == "User":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**AI:** {message}")

# Text input for user query
user_question = st.text_input("Ask a question about your workouts:")

# Handle question submission
if st.button("Submit"):
    if user_question:
        # Append user question to chat history
        st.session_state.chat_history.append(("User", user_question))

        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(user_question)

        # Combine the content of retrieved documents for context
        user_data = "\n".join([doc.page_content for doc in relevant_docs])

        # Prepare the input parameters for the prompt
        query_params = {
            "question": user_question,
            "user_data": user_data,  # Combined retrieved documents
            "persona": persona,
            "theme": theme
        }

        # Generate AI response with the LLM chain using the parameters
        ai_response = llm_chain(query_params)

        # Append AI response to chat history
        st.session_state.chat_history.append(("AI", ai_response['text']))

        # Display response and sources
        st.markdown(f"**AI Response:** {ai_response['text']}")
        st.markdown("**Sources:**")
        for doc in relevant_docs:
            st.markdown(f"- {doc.page_content}")

        # Clear the text input
        st.session_state.user_question = ""

# Clear conversation
if st.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.experimental_set_query_params()

# Display additional information
st.sidebar.markdown("### About the Chatbot")
st.sidebar.write(
    "This chatbot is designed to provide assistance on fitness and workout routines, integrating advanced RAG techniques with the Ollama LLM.")
