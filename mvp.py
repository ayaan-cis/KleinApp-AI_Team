from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st

# Define the prompt template
template = """
You are a training assistant for a Strength Training App that answers questions for app users.

Ground your response in factual data from your pre-training set,
specifically referencing or quoting authoritative sources when possible.
Respond to this question using information that can be attributed to the USER DATA.

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