import pickle
import streamlit as st

model = pickle.load(open('model.pkl', 'rb'))

# Define a function for making predictions
def predict_leave(input_data):
    prediction = model.predict([input_data])[0]
    return "will go on leave" if prediction == 1 else "will not go on leave"

# Create a Streamlit app
st.title('HR Leave Solution')

st.write('Welcome to the HR Solution Web App! Enter employee details to predict leave status.')

# Input fields for employee details
joining_year = st.slider('Joining Year', 2000, 2030, 2010)
payment_tier = st.selectbox('Payment Tier', [1, 2, 3])
age = st.slider('Age', 18, 80, 30)
experience = st.slider('Experience (years)', 0, 40, 5)
edu_rank = st.slider('Education Rank', 1, 5, 3)
ever_benched = st.radio('Ever Benched', ('No', 'Yes'))
gender = st.radio('Gender', ('Female', 'Male'))
age_range_youths = st.radio('Age Range (Youths)', ('No', 'Yes'))

# Convert radio button selections to 0 or 1
ever_benched = 1 if ever_benched == 'Yes' else 0
gender = 1 if gender == 'Male' else 0
age_range_youths = 1 if age_range_youths == 'Yes' else 0

# Create a button to make a prediction
if st.button('Predict'):
    input_data = [joining_year, payment_tier, age, experience,
                  edu_rank, ever_benched, gender, age_range_youths]
    prediction = predict_leave(input_data)
    st.write(f'The employee {prediction}.')
