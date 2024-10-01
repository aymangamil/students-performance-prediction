import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# Load model and scaler
with open("C:/Users/Ayman/Downloads/saved_linear_model_new.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open("C:/Users/Ayman/Downloads/saved_scaler_linear_new.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Set page config
st.set_page_config(page_title="Student Performance Prediction", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-image: url('C:/Users/Ayman/Downloads/WhatsApp Image 2024-09-24 at 16.27.31_0cfd7f07.jpg');
        background-size: cover;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s, color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white; /* Keep text white on hover */
    }
    .input-section, .prediction-section, .visualization-section {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 20px;
        color: white;
    }
    .feature-icon {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Student Performance Prediction")
st.image("C:/Users/Ayman/Downloads/R (6).jpeg", use_column_width=True)

# Input Section
st.header("Enter Student Details:")
st.markdown("<div class='input-section'>", unsafe_allow_html=True)

# Collecting user input
hours_studied = st.number_input("Hours Studied:", min_value=0.0)
attendance = st.number_input("Attendance:", min_value=0.0)
sleep_hours = st.number_input("Sleep Hours:", min_value=0.0)
previous_scores = st.number_input("Previous Scores:", min_value=0.0)
tutoring_sessions = st.number_input("Tutoring Sessions:", min_value=0.0)
physical_activity = st.number_input("Physical Activity:", min_value=0.0)

parental_involvement = st.selectbox("Parental Involvement:", ["Low", "Medium", "High"])
access_to_resources = st.selectbox("Access to Resources:", ["Low", "Medium", "High"])
extracurricular_activities = st.selectbox("Extracurricular Activities:", ["Yes", "No"])
motivation_level = st.selectbox("Motivation Level:", ["Low", "Medium", "High"])
internet_access = st.selectbox("Internet Access:", ["Yes", "No"])
family_income = st.selectbox("Family Income:", ["Low", "Medium", "High"])
teacher_quality = st.selectbox("Teacher Quality:", ["Low", "Medium", "High"])
school_type = st.selectbox("School Type:", ["Public", "Private"])
peer_influence = st.selectbox("Peer Influence:", ["Neutral", "Positive", "Negative"])
learning_disabilities = st.selectbox("Learning Disabilities:", ["Yes", "No"])
parental_education_level = st.selectbox("Parental Education Level:", ["High School", "Undergraduate", "Postgraduate"])
distance_from_home = st.selectbox("Distance from Home:", ["Near", "Moderate", "Far"])
gender = st.selectbox("Gender:", ["Male", "Female"])

st.markdown("</div>", unsafe_allow_html=True)

# Prediction Section
st.header("Prediction")
st.markdown("<div class='prediction-section'>", unsafe_allow_html=True)

predicted_exam_score = None  # Initialize the predicted_exam_score variable

# Submit button
if st.button("Submit"):
    try:
        # Convert inputs to features list
        features = [
            hours_studied, attendance, sleep_hours, previous_scores, tutoring_sessions, physical_activity,
            1 if parental_involvement == "Low" else 0,
            1 if parental_involvement == "Medium" else 0,
            1 if access_to_resources == "Low" else 0,
            1 if access_to_resources == "Medium" else 0,
            1 if extracurricular_activities == "Yes" else 0,
            1 if motivation_level == "Low" else 0,
            1 if motivation_level == "Medium" else 0,
            1 if internet_access == "Yes" else 0,
            1 if family_income == "Low" else 0,
            1 if family_income == "Medium" else 0,
            1 if teacher_quality == "Low" else 0,
            1 if teacher_quality == "Medium" else 0,
            1 if school_type == "Public" else 0,
            1 if peer_influence == "Neutral" else 0,
            1 if peer_influence == "Positive" else 0,
            1 if learning_disabilities == "Yes" else 0,
            1 if parental_education_level == "High School" else 0,
            1 if parental_education_level == "Postgraduate" else 0,
            1 if distance_from_home == "Moderate" else 0,
            1 if distance_from_home == "Near" else 0,
            1 if gender == "Male" else 0
        ]

        features_array = np.array([features])

        # Scale features and make prediction
        features_scaled = scaler.transform(features_array)
        predicted_exam_score = model.predict(features_scaled)

        st.success(f"Predicted Exam Score: {predicted_exam_score[0]:.2f}")

        # Celebratory effects if score is >= 50
        if predicted_exam_score[0] >= 50:
            st.balloons()
        else:
            st.error("The predicted score is below 50. Please consider improving!")
            st.snow()  # Snow effect to simulate sadness

    except ValueError as e:
        st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)

# Visualization Section
st.header("Visualizations")
st.markdown("<div class='visualization-section'>", unsafe_allow_html=True)

if predicted_exam_score is not None:
    # Donut chart of predicted score
    fig = go.Figure(data=[go.Pie(
        labels=["Predicted Score", "Remaining Score"],
        values=[predicted_exam_score[0], 100 - predicted_exam_score[0]],
        hole=0.6,
        marker=dict(colors=['red', '#D2B48C']),
    )])
    fig.update_layout(title_text='Predicted Exam Score Distribution')
    st.plotly_chart(fig)

    # Feature Icons
    feature_icons = {
        "Sleep Hours": (r"C:\Users\Ayman\Pictures\Screenshots\Screenshot 2024-10-02 005621.png", sleep_hours, 6, 8),
        "Hours Studied": (r"C:\Users\Ayman\Pictures\Screenshots\Screenshot 2024-10-02 005644.png", hours_studied, 2, 5),
        "Attendance": (r"C:\Users\Ayman\Pictures\Screenshots\Screenshot 2024-10-02 005720.png", attendance, 75, 90),
        # Add other features here with corresponding icons and thresholds
    }

    for feature_name, (icon_path, value, low_threshold, high_threshold) in feature_icons.items():
        icon = Image.open(icon_path)
        st.markdown("<div class='feature-icon'>", unsafe_allow_html=True)
        st.image(icon, width=50)
        st.write(f"{feature_name}: {value}")

        # Indicate whether the value is low, moderate, or good
        if value < low_threshold:
            st.markdown("<span style='color: red;'>Low</span>", unsafe_allow_html=True)
        elif low_threshold <= value <= high_threshold:
            st.markdown("<span style='color: orange;'>Moderate</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: green;'>Good</span>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Display advice for students
st.header("Advice for you")

if predicted_exam_score is not None:
    st.markdown("<div class='visualization-section'>", unsafe_allow_html=True)

    if predicted_exam_score[0] >= 50:
        st.success("Congratulations! You have passed. Here are some tips to further improve:")
        st.write("- Continue to maintain a good study schedule.")
        st.write("- Stay consistent with your attendance and physical activities.")
        st.write("- Keep your motivation levels high and seek help when needed.")
    else:
        st.warning("Dear student, it seems you did not achieve the score you hoped for. Don't lose hope! Here are some tips to improve your performance in the future:")
        st.write("- Study regularly and focus on the subjects you need to improve.")
        st.write("- Try to understand your weaknesses and seek help from teachers or friends.")
        st.write("- Use mistakes as learning opportunities and don’t hesitate to try again.")
        st.write("- Remember that success requires patience and perseverance, and every step brings you closer to your goal.")
        st.write("- Organize your time better and stay motivated to achieve your dreams.")
st.markdown("<div style='text-align: center; color: black; font-style: italic;'>elmano</div>", unsafe_allow_html=True)    












