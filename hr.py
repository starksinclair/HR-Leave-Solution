import pickle
import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# df = pd.read_csv("Employee.csv")
# # Create new column 'Age Range'
# df['Age_range'] = df["Age"].apply(lambda x: "Early Youth" if x < 31 else "Youths")
# df = pd.get_dummies(df, columns = ['EverBenched', 'Gender','Age_range'],
#                      #prefix = ['EverBenched', 'Gender','Age_range'],
#                      drop_first = True, dtype = int)

# # Split df into X features and y target
# X = df.drop(['LeaveOrNot', 'Education', 'City'], axis = 1)
# y = df['LeaveOrNot']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)

model = pickle.load(open('model.pkl', 'rb'))

# st.set_page_config(
#     page_title='Human Resources Leave Prediction Solution',
#     layout='wide',
#     initial_sidebar_state='expanded'
# )

# Define a function for making predictions
def predict_leave(input_data):
    prediction = model.predict([input_data])[0]
    return "will go on leave" if prediction == 1 else "will not go on leave"

# Create a Streamlit app title
st.title('HR Leave Solution')

st.write('Welcome to the HR Solution Web App! Enter employee details to predict leave status.')

# Input fields for employee details
joining_year = st.slider('Joined Year', 2000, 2023, 2010)
payment_tier = st.selectbox('Payment Tier', [1, 2, 3])
age = st.slider('Age', 18, 60, 20)
experience = st.slider('Experience (years)', 0, 35, 5)
edu_rank = st.slider('Education Rank', 1, 5, 3)
ever_benched = st.radio('Ever Been Idle', ('No', 'Yes'))
gender = st.radio('Gender', ('Female', 'Male'))
age_range_youths = st.radio('Youth?', ('No', 'Yes'))

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

#st.subheader('Feature Importance')
# importance = rf.feature_importances_
# feature_names = df.columns
# plt.barh(feature_names, importance)
# st.pyplot()