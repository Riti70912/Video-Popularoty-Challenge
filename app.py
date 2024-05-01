import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import datetime
import pandas as pd
from datetime import datetime
import pickle

le = LabelEncoder()
model = pickle.load(open('modelTuned.pkl','rb'))

# Function to split date and time
def split_date_time(publishedAt):
    time = publishedAt.hour
    return time

# Function to extract hour
def extract_hour(time_obj):
    total_hours = time_obj.hour + time_obj.minute / 60
    return total_hours

# Function to extract the day of the week
def extract_day_of_week(date_string):
    # Convert the date string to a datetime object
    date_object = datetime.strptime(str(date_string), '%Y-%m-%d')
    day_of_week = date_object.weekday()
    day_name = date_object.strftime('%A')
    return day_of_week, day_name

def hour_to_sec(hour):
    try:
        hour = str(hour.time())
    except AttributeError:
        hour = '00:00:00'  # Assign a default time if it's a date object
    hours, minutes, seconds = map(int, hour.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds



title = st.text_input("Title")
tags = st.text_input("Tags")
published_time = st.date_input("Published time") #datetime.date or a tuple with 0-2 dates or None
videoLength = st.time_input("videoLength") #returns datetime.time or None
commentCount = st.number_input("comment_count")
# viewCount = st.number_input("View count")
# likeCount = st.number_input("like count")

def get_weekday(date_obj):
    date_str = date_obj.strftime('%Y-%m-%d')
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    day_of_week_number = date_obj.weekday()
    return day_of_week_number

def features():
    feature = pd.DataFrame({
        "title": [title],
        "tags": [tags],
        "published_time": [published_time],
        "videoLength": [videoLength],
        "commentCount": [commentCount],
    })
    feature["title"] = le.fit_transform(feature['title'])
    feature["tags"] = le.fit_transform(feature["tags"])

    cols = ['commentCount']
    feature[cols] = feature[cols].apply(pd.to_numeric, errors='coerce', axis=1)

    feature["day_of_week_numeric"] = feature["published_time"].apply(get_weekday)
    feature['published_hour'] = feature['videoLength'].apply(split_date_time)
    feature["duration_seconds"] = feature["published_hour"].apply(hour_to_sec)  # Use published_hour instead of published_time

    # Drop datetime columns as they may cause data type conflicts
    feature.drop(["published_time"], axis=1, inplace=True)

    tuned_features = ['title','tags', 'published_hour', 'duration_seconds','day_of_week_numeric','commentCount']
    data = feature[tuned_features]
    return data



if st.button("Predict"):
    df = features()
    if model.predict(df)==1:
        st.write("Popular")
    else:
        st.write("Not popular")
