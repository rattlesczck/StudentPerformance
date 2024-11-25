# StudentPerformance
 Student Performance Prediction API . This project predicts a student's 5th semester marks based on their scores in the first four semesters. The model is built using a Random Forest Regressor. 
 # Model Choice
 The Random Forest Regressor was selected due to its robustness and ability to handle complex data with non-linear relationships.
 
## Installation Instructions
1. Clone this repository.
2. Install dependencies using:
   pip install -r requirements.txt

run the flask app (app.py)

## API Testing:
You can test the API using Postman by sending a POST request to  "http://127.0.0.1:5000/predict"  with the following JSON body:
eg :
   {
    "1st": 7.0,
    "2nd": 6.5,
    "3rd": 7.2,
    "4th": 8.0
}


The response should look like :
{
    "Predicted_5th_Mark": 7.50
}
