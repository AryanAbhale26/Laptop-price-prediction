from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and dataframe
df = pd.read_csv("df.csv")
pipe = pickle.load(open("pipe.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', company_options=df['Company'].unique(),
                           lap_type_options=df['TypeName'].unique(),
                           resolution_options=['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                               '2880x1800', '2560x1600', '2560x1440', '2304x1440'],
                           cpu_options=df['Cpu_brand'].unique(),
                           gpu_options=df['Gpu_brand'].unique(),
                           os_options=df['os'].unique())

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        company = request.form['company']
        lap_type = request.form['lap_type']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
        ips = 1 if request.form['ips'] == 'Yes' else 0
        screen_size = float(request.form['screen_size'])
        resolution = request.form['resolution']
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']
        
        # Calculate ppi
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
        
        # Prepare the query array
        query = np.array([company, lap_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)
        
        # Get the prediction using the model
        prediction = str(int(np.exp(pipe.predict(query)[0])))
        
        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction)

        # Your prediction code here
        
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
