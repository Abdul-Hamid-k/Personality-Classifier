import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
# from pathlib import Path
# import sys
# sys.path.append('../')
# from config import ROOT

# Load model, scaler, and encoder
with open('./PersonalityClassifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('./Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('./Encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

st.set_page_config(page_title="Personality Classifier", page_icon=":guardsman:", layout="wide")
st.title("🧠 Personality Type Prediction App")
st.write("Answer questions about your personality traits to discover your personality type!")
st.write("---")

# Feature list (all features except personality_type)
features = [
    'social_energy', 'alone_time_preference', 'talkativeness', 'deep_reflection',
    'group_comfort', 'party_liking', 'listening_skill', 'empathy', 'creativity',
    'organization', 'leadership', 'risk_taking', 'public_speaking_comfort',
    'curiosity', 'routine_preference', 'excitement_seeking', 'friendliness',
    'emotional_stability', 'planning', 'spontaneity', 'adventurousness',
    'reading_habit', 'sports_interest', 'online_social_usage', 'travel_desire',
    'gadget_usage', 'work_style_collaborative', 'decision_speed', 'stress_handling'
]

# Create input form in columns
col1, col2 = st.columns(2)

user_input = {}

# Display sliders in two columns
with col1:
    st.subheader("Social Traits")
    user_input['social_energy'] = st.slider("Social Energy", 0, 10, 5, help="How energized do you feel in social situations?")
    user_input['alone_time_preference'] = st.slider("Alone Time Preference", 0, 10, 5, help="How much do you prefer being alone?")
    user_input['talkativeness'] = st.slider("Talkativeness", 0, 10, 5, help="How talkative are you?")
    user_input['listening_skill'] = st.slider("Listening Skill", 0, 10, 5, help="How well do you listen?")
    user_input['empathy'] = st.slider("Empathy", 0, 10, 5, help="How empathetic are you?")
    user_input['friendliness'] = st.slider("Friendliness", 0, 10, 5, help="How friendly are you?")
    user_input['group_comfort'] = st.slider("Group Comfort", 0, 10, 5, help="How comfortable are you in groups?")
    user_input['party_liking'] = st.slider("Party Liking", 0, 10, 5, help="Do you enjoy parties?")
    user_input['public_speaking_comfort'] = st.slider("Public Speaking Comfort", 0, 10, 5, help="How comfortable with public speaking?")
    user_input['online_social_usage'] = st.slider("Online Social Usage", 0, 10, 5, help="How much do you use social media?")

with col2:
    st.subheader("Cognitive & Behavioral Traits")
    user_input['deep_reflection'] = st.slider("Deep Reflection", 0, 10, 5, help="Do you reflect deeply on things?")
    user_input['creativity'] = st.slider("Creativity", 0, 10, 5, help="How creative are you?")
    user_input['organization'] = st.slider("Organization", 0, 10, 5, help="How organized are you?")
    user_input['leadership'] = st.slider("Leadership", 0, 10, 5, help="How strong are your leadership skills?")
    user_input['risk_taking'] = st.slider("Risk Taking", 0, 10, 5, help="How willing are you to take risks?")
    user_input['curiosity'] = st.slider("Curiosity", 0, 10, 5, help="How curious are you?")
    user_input['routine_preference'] = st.slider("Routine Preference", 0, 10, 5, help="How much do you prefer routines?")
    user_input['excitement_seeking'] = st.slider("Excitement Seeking", 0, 10, 5, help="Do you seek excitement?")
    user_input['planning'] = st.slider("Planning", 0, 10, 5, help="How much do you plan ahead?")
    user_input['spontaneity'] = st.slider("Spontaneity", 0, 10, 5, help="How spontaneous are you?")

st.subheader("Interests & Preferences")
col3, col4, col5 = st.columns(3)

with col3:
    user_input['adventurousness'] = st.slider("Adventurousness", 0, 10, 5, help="How adventurous are you?")
    user_input['reading_habit'] = st.slider("Reading Habit", 0, 10, 5, help="How much do you read?")

with col4:
    user_input['sports_interest'] = st.slider("Sports Interest", 0, 10, 5, help="How interested in sports?")
    user_input['gadget_usage'] = st.slider("Gadget Usage", 0, 10, 5, help="How much do you use gadgets?")

with col5:
    user_input['travel_desire'] = st.slider("Travel Desire", 0, 10, 5, help="How much do you want to travel?")
    user_input['work_style_collaborative'] = st.slider("Collaborative Work Style", 0, 10, 5, help="Prefer collaborative work?")

user_input['emotional_stability'] = st.slider("Emotional Stability", 0, 10, 5, help="How emotionally stable are you?")
user_input['decision_speed'] = st.slider("Decision Speed", 0, 10, 5, help="How fast do you make decisions?")
user_input['stress_handling'] = st.slider("Stress Handling", 0, 10, 5, help="How well do you handle stress?")

# Prediction button
st.write("---")
if st.button("🔮 Predict Personality Type", use_container_width=True):
    # Prepare input data
    input_df = pd.DataFrame([user_input])
    input_df = input_df[features]  # Ensure correct column order
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)
    
    # Decode the prediction
    personality_type = encoder.inverse_transform([prediction])[0]
    
    # Display results
    st.success(f"### Your Personality Type: **{personality_type}**")
    
    # Show confidence
    confidence = max(prediction_proba[0]) * 100
    st.info(f"Confidence: {confidence:.2f}%")
    
    # Show all probabilities
    st.subheader("Personality Type Probabilities")
    prob_df = pd.DataFrame({
        'Personality Type': encoder.inverse_transform(range(len(prediction_proba[0]))),
        'Probability': prediction_proba[0]
    }).sort_values('Probability', ascending=False)
    
    st.bar_chart(prob_df.set_index('Personality Type'))


