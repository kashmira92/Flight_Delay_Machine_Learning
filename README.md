
# Flight Delay Prediction

## Overview
This project aims to predict flight delays using a machine learning model. The prediction model is deployed as a web application using Streamlit, allowing users to input flight details and get delay predictions.

## Features
- **User Inputs**: 
  - Flight Date
  - Airline
  - Origin Airport
  - Destination Airport
  - Scheduled Departure Time
  - Scheduled Arrival Time
- **Prediction**: 
  - Predicts whether a flight will be delayed based on user inputs.
- **Results**: 
  - Displays whether the flight is on time or delayed.

## Dataset
The dataset used for training the model includes the following attributes:
- `FlightDate`: The date of the flight.
- `Airline`: The airline operating the flight.
- `Origin`: The airport code for the origin airport.
- `Dest`: The airport code for the destination airport.
- `CRSDepTime`: The scheduled departure time.
- `CRSArrTime`: The scheduled arrival time.
- And various other attributes related to flight scheduling and performance.

## Model
- **Algorithm**: Random Forest
- **Training**: The model was trained on historical flight data with extensive feature engineering and preprocessing steps.

## Setup

### Prerequisites
- Python 3.x
- Required libraries: Streamlit, Pandas, Scikit-learn, Joblib, Matplotlib, Seaborn

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/flight-delay-prediction.git
   cd flight-delay-prediction
   ```
2. Install the required libraries:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Application
1. Navigate to the project directory.
2. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit app in your browser.
2. Fill in the flight details (Flight Date, Airline, Origin, Destination, Scheduled Departure Time, Scheduled Arrival Time).
3. Click the "Submit" button to get the prediction.
4. The application will display whether the flight is predicted to be on time or delayed.
