from flask import Flask, request, jsonify, render_template, url_for
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data/MY2019 Fuel Consumption Ratings.csv")

# Label Encoding
le_make = LabelEncoder()
df['Make'] = le_make.fit_transform(df['Make'])

le_fuel_type = LabelEncoder()
df['Fuel_Type'] = le_fuel_type.fit_transform(df['Fuel_Type'])

# Converting Pandas DataFrame into Numpy array
X = df[['Engine_Size (L)', 'Make', 'Fuel_Type', 'Cylinders', 'Fuel_Consumption_City', 'Fuel_Consumption_Hwy']] .values
y = df[['CO2_Emissions']] .values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    engine = request.form['engine']
    make = request.form['make']
    fuel = request.form['fuel']
    cylinders = request.form['cylinders']
    Fuel_Consumption_City = request.form['Fuel_Consumption_City']
    Fuel_Consumption_Hwy = request.form['Fuel_Consumption_Hwy']
    
    x1 = [engine, make, fuel, cylinders, Fuel_Consumption_City, Fuel_Consumption_Hwy]
    df = pd.DataFrame(data=[x1], columns=['Engine_Size (L)', 'Make', 'Fuel_Type', 'Cylinders', 'Fuel_Consumption_City', 'Fuel_Consumption_Hwy'])
    
    df['Make'] = le_make.transform(df['Make'])
    df['Fuel_Type'] = le_fuel_type.transform(df['Fuel_Type'])
    
    X = df.iloc[:, :6].values
    ans = model.predict(X)
    output = ans
    
    return render_template('index.html', prediction_text='Fuel Level: {}'.format(output))

if __name__ == 'main':
    app.run(debug=False)
