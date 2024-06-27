import streamlit as st
import pandas as pd
# import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# filename = 'final_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# filename = 'random_forest_model_wh.pkl'
# joblib_file = "smote_random_forest_model_wh.joblib"
joblib_file = "random_forest_30M_wh.joblib"
import joblib
# loaded_model = joblib.load(filename)
loaded_model = joblib.load(joblib_file)
# df = pd.read_csv("flight_cleaned_rf_wh.csv")
# df = pd.read_csv("flight_cleaned.csv")
# df = df.drop(columns=['FlightDate'])
# df = pd.read_csv("flight_new_features.csv")
df = pd.read_csv("flight_new_features1.csv")


st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("Flight Delay Prediction")

with st.form("my_form"):
    FlightDate = st.date_input(label='FlightDate', help='Format: YYYY-MM-DD')
    Airline = st.selectbox('Airline', options=df['Airline'].unique())
    Origin = st.selectbox('Origin', options=df['Origin'].unique())
    Dest = st.selectbox('Dest', options=df['Dest'].unique())
    CRSDepTime = st.number_input(label='CRSDepTime', step=1, format="%d")
    CRSArrTime = st.number_input(label='CRSArrTime', step=1, format="%d")
    # Year = st.number_input(label='Year', step=1, format="%d")
    # Month = st.number_input(label='Month', step=1, min_value=1, max_value=12, format="%d")
    # DayofMonth = st.number_input(label='DayofMonth', step=1, min_value=1, max_value=31, format="%d")
    
    
    df = df.drop(columns=['FlightDate', 'CRSDepHour', 'ArrDelay','Year'])


    # Encode categorical variables
    label_encoders = {}
    for col in ['Airline', 'Origin', 'Dest', 'time_of_day', 'season']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split data into features and target
    # X = df.drop(['ArrDelay', 'Delayed'], axis=1)
    X = df.drop(['Delayed'], axis=1)
    y = df['Delayed']
    

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
     
    submitted = st.form_submit_button("Submit")

if submitted:

    new_data = pd.DataFrame({
        'FlightDate':[FlightDate],
        'Airline': [Airline],
        'Origin': [Origin],
        'Dest': [Dest],
        'CRSDepTime': [CRSDepTime],
        'CRSArrTime': [CRSArrTime]
    })

    #preprocessing
    # Convert FlightDate to datetime
    new_data['FlightDate'] = pd.to_datetime(new_data['FlightDate'])
    
    # Extract the year, month, and day of month from 'FlightDate'
    # new_data['Year'] = new_data['FlightDate'].dt.year
    new_data['Month'] = new_data['FlightDate'].dt.month
    new_data['DayofMonth'] = new_data['FlightDate'].dt.day

    # Create 'is_weekend' feature
    # data['is_weekend'] = data['FlightDate'].dt.dayofweek >= 5
    new_data['is_weekend'] = (new_data['FlightDate'].dt.dayofweek >= 5).astype(int)

    from pandas.tseries.holiday import USFederalHolidayCalendar
    # Is it a holiday?
    # Create 'is_holiday' feature using USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=new_data['FlightDate'].min(), end=new_data['FlightDate'].max())
    # data['is_holiday'] = data['FlightDate'].isin(holidays)

    new_data['is_holiday'] = new_data['FlightDate'].isin(holidays).astype(int)

    # Create 'DayOfWeek' feature
    # Monday=0>>Sunday=6
    new_data['DayOfWeek'] = new_data['FlightDate'].dt.dayofweek


    # Create 'time_of_day' feature
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

    new_data['CRSDepHour'] = new_data['CRSDepTime'] // 100
    new_data['time_of_day'] = new_data['CRSDepHour'].apply(get_time_of_day)

    # Create 'season' feature
    def get_season(month):
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

    new_data['season'] = new_data['Month'].apply(get_season)
    
    new_data = new_data.drop(columns=['FlightDate', 'CRSDepHour'])
    
    # Apply label encoding
    for col in ['Airline', 'Origin', 'Dest', 'time_of_day', 'season']:
        le = label_encoders[col]
        new_data[col] = le.transform(new_data[col])

    #test the data
    # st.write(new_data)

    # Convert to numpy array and scale the data
    new_data_array = new_data.values
    new_data_scaled = scaler.transform(new_data_array)

    # Make prediction
    prediction = loaded_model.predict(new_data_scaled)
    # Print the result based on the prediction
    if prediction[0] == 0:
        result = "The flight is on time"
    else:
        result = "The flight is delayed"
    st.write(result)
    st.write(prediction[0])
###################################################
    # print(result)
    # return "Delayed" if prediction[0] else "On Time"
    # pred = loaded_model.predict(data)[0]
    # st.write('Flight prediction:', pred)

# with st.form("my_form"):
#     Airline=st.number_input(label='Airline',step=0.001,format="%.6f")
#     Origin=st.number_input(label='Origin',step=0.001,format="%.6f")
#     Dest=st.number_input(label='Dest',step=0.01,format="%.2f")
#     CRSDepTime=st.number_input(label='CRSDepTime',step=0.01,format="%.2f")
#     CRSArrTime=st.number_input(label='CRSArrTime',step=0.01,format="%.2f")
#     Year=st.number_input(label='Year',step=0.01,format="%.6f")
#     Month=st.number_input(label='Month',step=0.01,format="%.6f")
#     DayofMonth=st.number_input(label='DayofMonth',step=0.1,format="%.6f")
#     data=[[Airline,Origin,Dest,CRSDepTime,CRSArrTime,Year,Month,DayofMonth]]

#     submitted = st.form_submit_button("Submit")

# if submitted:
#     pred=loaded_model.predict(data)[0]
#     print('Flight is..',pred)

