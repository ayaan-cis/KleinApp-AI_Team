from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
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

# Inject custom CSS for styling
with open("Klein App/chatbot_styles.css") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Define the prompt template
template ="""
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

# Set up the prompt and model
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3")
chain = prompt | model

# Example user data
user_data = {
    "workouts": [
        {
            "date": "2024-11-01",
            "type": "Upper Body Strength",
            "exercises": [
                {
                    "name": "Bench Press",
                    "muscleGroup": "Chest",
                    "equipment": "Barbell",
                    "sets": [
                        {"reps": 10, "weight": 135},
                        {"reps": 8, "weight": 155},
                        {"reps": 6, "weight": 175}
                    ]
                },
                {
                    "name": "Pull-Up",
                    "muscleGroup": "Back",
                    "equipment": "Bodyweight",
                    "sets": [
                        {"reps": 10, "weight": "Bodyweight"},
                        {"reps": 8, "weight": "Bodyweight + 10 lbs"},
                        {"reps": 6, "weight": "Bodyweight + 15 lbs"}
                    ]
                }
            ]
        },
        {
            "date": "2024-11-02",
            "type": "Lower Body Strength",
            "exercises": [
                {
                    "name": "Squat",
                    "muscleGroup": "Legs",
                    "equipment": "Barbell",
                    "sets": [
                        {"reps": 10, "weight": 185},
                        {"reps": 8, "weight": 205},
                        {"reps": 6, "weight": 225}
                    ]
                },
                {
                    "name": "Leg Press",
                    "muscleGroup": "Legs",
                    "equipment": "Machine",
                    "sets": [
                        {"reps": 12, "weight": 300},
                        {"reps": 10, "weight": 350},
                        {"reps": 8, "weight": 400}
                    ]
                }
            ]
        },
        {
            "date": "2024-11-03",
            "type": "Full Body Strength",
            "exercises": [
                {
                    "name": "Deadlift",
                    "muscleGroup": "Back, Legs",
                    "equipment": "Barbell",
                    "sets": [
                        {"reps": 5, "weight": 225},
                        {"reps": 5, "weight": 245},
                        {"reps": 5, "weight": 265}
                    ]
                },
                {
                    "name": "Overhead Press",
                    "muscleGroup": "Shoulders",
                    "equipment": "Barbell",
                    "sets": [
                        {"reps": 10, "weight": 95},
                        {"reps": 8, "weight": 105},
                        {"reps": 6, "weight": 115}
                    ]
                }
            ]
        }
    ]
}

# Initialize Streamlit session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = "What are my workouts for my recent training session?"

# Display the conversation history
st.title("Strength Training Assistant")
for message in st.session_state.conversation:
    st.write(message)

# Function to invoke the model and update the conversation
def ask_question(question, user_data):
    # Generate the response from the model
    response = chain.invoke({"question": question, "user_data": user_data})
    # Store the question and response in conversation history
    st.session_state.conversation.append(f"**You:** {question}")
    st.session_state.conversation.append(f"**Assistant:** {response}")
    # Display the response immediately
    st.write(f"**Assistant:** {response}")

# Input area for new questions
user_question = st.text_input("Ask another question:", value=st.session_state.current_question)

# Handle button click to submit the question and get a response
if st.button("Ask"):
    ask_question(user_question, user_data)
    # Clear the input field for the next interaction
    st.session_state.current_question = ""

# Option to reset the conversation
if st.button("Reset Conversation"):
    st.session_state.conversation = []
    st.session_state.current_question = "What are my workouts for my recent training session?"


