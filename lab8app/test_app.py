import requests

input = {
    "Weight_kg": 54.0,
    "Age": "20-25",
    "Hyperandrogenism": "No",
    "Hirsutism": "No",
    "Conception_Difficulty": "No",
    "Insulin_Resistance": "No",
    "Exercise_Frequency": "Rarely",
    "Exercise_Type": "No Exercise",
    "Exercise_Duration": "Less than 30 minutes",
    "Sleep_Hours": "6-8 hours",
    "Exercise_Benefit": "Somewhat",
    "Hormonal_Imbalance": "No"
    }

url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=input)
print(response.json())
