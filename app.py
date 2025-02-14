from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training completed"

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            delivery_person_id = request.form['Delivery_person_ID']
            delivery_person_age = float(request.form['Delivery_person_Age'])
            delivery_person_ratings = float(request.form['Delivery_person_Ratings'])
            restaurant_latitude = float(request.form['Restaurant_latitude'])
            restaurant_longitude = float(request.form['Restaurant_longitude'])
            delivery_location_latitude = float(request.form['Delivery_location_latitude'])
            delivery_location_longitude = float(request.form['Delivery_location_longitude'])
            type_of_order = request.form['Type_of_order']
            type_of_vehicle = request.form['Type_of_vehicle']
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            precipitation = float(request.form['precipitation'])
            weather_description = request.form['weather_description']
            traffic_level = request.form['Traffic_Level']
            distance = float(request.form['Distance'])

            # Create a DataFrame with proper column names
            import pandas as pd
            input_df = pd.DataFrame([{
                "Delivery_person_ID": delivery_person_id,
                "Delivery_person_Age": delivery_person_age,
                "Delivery_person_Ratings": delivery_person_ratings,
                "Restaurant_latitude": restaurant_latitude,
                "Restaurant_longitude": restaurant_longitude,
                "Delivery_location_latitude": delivery_location_latitude,
                "Delivery_location_longitude": delivery_location_longitude,
                "Type_of_order": type_of_order,
                "Type_of_vehicle": type_of_vehicle,
                "temperature": temperature,
                "humidity": humidity,
                "precipitation": precipitation,
                "weather_description": weather_description,
                "Traffic_Level": traffic_level,
                "Distance (km)": distance
            }])
            
            obj = PredictionPipeline()
            prediction = obj.predict(input_df)
            
            return render_template('results.html', prediction=str(prediction))
        except Exception as e:
            print('The Exception message is:', e)
            return 'Something went wrong'
    else:
        return render_template('index.html')
    
if __name__ == "__main__":
	
	app.run(host="0.0.0.0", port = 8080)

