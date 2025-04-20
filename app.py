from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import random

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model.pkl')
label_encoders = joblib.load('encoders.pkl')
df = pd.read_csv('processed_data.csv')

# Columns to encode
categorical_cols = ['Gender', 'Activity_Level', 'Diet_Type', 'Health_Condition']

# Food recommendation database
food_recommendations = {
    "Omnivore": {
        "None": [
            "Lean protein (chicken, turkey, lean beef), eggs, dairy, whole grains (brown rice, quinoa), colorful vegetables, fruits, nuts, and seeds.",
            "Salmon, sweet potatoes, oatmeal, Greek yogurt, spinach, berries, avocados, almonds, and olive oil."
        ],
        "Diabetes": [
            "Low glycemic foods: lean proteins, non-starchy vegetables, whole grains, nuts, and berries.",
            "Salmon, chicken, eggs, broccoli, spinach, beans, lentils, quinoa, and small portions of berries."
        ],
        "Hypertension": [
            "DASH diet: fruits, vegetables, whole grains, lean proteins, low-fat dairy, and limited sodium.",
            "Bananas, spinach, beets, oats, yogurt, berries, lean poultry, and unsalted nuts."
        ],
        "Anemia": [
            "Iron-rich foods: lean red meat, dark leafy greens, beans, fortified cereals with vitamin C sources.",
            "Lean beef, spinach, lentils, quinoa, tofu, vitamin C rich fruits, and iron-fortified cereals."
        ]
    },
    "Vegetarian": {
        "None": [
            "Eggs, dairy, legumes, tofu, tempeh, seitan, whole grains, nuts, seeds, and plenty of vegetables.",
            "Greek yogurt, cottage cheese, lentils, chickpeas, quinoa, almonds, chia seeds, and vegetables."
        ],
        "Diabetes": [
            "Low glycemic plant proteins, eggs, dairy, non-starchy vegetables, nuts, seeds, and limited fruits.",
            "Eggs, tofu, beans, lentils, broccoli, spinach, nuts, and small portions of berries."
        ],
        "Hypertension": [
            "DASH diet adapted: eggs, low-fat dairy, beans, vegetables, fruits, whole grains, and limited sodium.",
            "Oats, low-fat dairy, beans, leafy greens, berries, bananas, and unsalted nuts."
        ],
        "Anemia": [
            "Iron-rich plant foods, eggs, vitamin C sources, iron-fortified foods.",
            "Eggs, spinach, lentils, fortified cereals, tofu, beans, vitamin C rich fruits."
        ]
    },
    "Vegan": {
        "None": [
            "Legumes, tofu, tempeh, seitan, whole grains, nutritional yeast, nuts, seeds, and abundant vegetables.",
            "Tofu, lentils, chickpeas, quinoa, chia seeds, nutritional yeast, hemp seeds, and varied vegetables."
        ],
        "Diabetes": [
            "Low glycemic plant proteins, non-starchy vegetables, nuts, seeds, and limited fruits.",
            "Tofu, beans, lentils, broccoli, spinach, nuts, and small portions of berries."
        ],
        "Hypertension": [
            "Plant-based DASH approach: legumes, vegetables, fruits, whole grains, and limited sodium.",
            "Tofu, beans, leafy greens, berries, bananas, and unsalted nuts."
        ],
        "Anemia": [
            "Iron-rich plant foods, vitamin C sources, iron-fortified foods, and supplements if needed.",
            "Spinach, lentils, fortified cereals, tofu, beans, vitamin C rich fruits, and consider B12/iron supplements."
        ]
    },
    "Pescatarian": {
        "None": [
            "Fish, seafood, eggs, dairy, legumes, whole grains, nuts, seeds, and plenty of vegetables.",
            "Salmon, tuna, Greek yogurt, eggs, lentils, quinoa, walnuts, and colorful vegetables."
        ],
        "Diabetes": [
            "Fatty fish, eggs, dairy, legumes, non-starchy vegetables, nuts, seeds, and limited fruits.",
            "Salmon, sardines, eggs, Greek yogurt, beans, leafy greens, nuts, and small portions of berries."
        ],
        "Hypertension": [
            "DASH diet with fish: fatty fish, eggs, low-fat dairy, beans, vegetables, fruits, whole grains, and limited sodium.",
            "Salmon, mackerel, low-fat dairy, beans, leafy greens, berries, bananas, and unsalted nuts."
        ],
        "Anemia": [
            "Iron-rich seafood, eggs, plant sources, vitamin C rich foods.",
            "Clams, oysters, tuna, eggs, spinach, lentils, vitamin C rich fruits."
        ]
    },
    "Keto": {
        "None": [
            "High-fat foods, moderate protein, very low carbs: meats, fatty fish, eggs, dairy, nuts, seeds, and low-carb vegetables.",
            "Avocados, olive oil, fatty fish, eggs, cheese, nuts, seeds, and leafy greens."
        ],
        "Diabetes": [
            "Keto-friendly proteins, healthy fats, non-starchy vegetables, and careful monitoring.",
            "Fatty fish, chicken, eggs, avocados, olive oil, nuts, and leafy greens."
        ],
        "Hypertension": [
            "Heart-healthy fats, adequate protein, non-starchy vegetables, and limited sodium.",
            "Avocados, olive oil, fatty fish, unsalted nuts, seeds, and potassium-rich low-carb vegetables."
        ],
        "Anemia": [
            "Iron-rich proteins, vitamin C rich low-carb foods, and possibly supplements.",
            "Red meat, eggs, spinach, broccoli with added lemon juice for vitamin C."
        ]
    },
    "Paleo": {
        "None": [
            "Meats, fish, eggs, vegetables, fruits, nuts, seeds, and healthy oils.",
            "Grass-fed beef, wild-caught salmon, eggs, sweet potatoes, berries, nuts, and olive oil."
        ],
        "Diabetes": [
            "Lean proteins, non-starchy vegetables, limited fruits, nuts, and seeds.",
            "Chicken, turkey, fish, eggs, leafy greens, nuts, and small portions of berries."
        ],
        "Hypertension": [
            "Lean proteins, vegetables, fruits, nuts, seeds, and limited sodium.",
            "Fish, poultry, vegetables, fruits, unsalted nuts, and limited dried fruits."
        ],
        "Anemia": [
            "Iron-rich meats, organ meats, vegetables, vitamin C sources.",
            "Lean red meat, liver, spinach, broccoli, and vitamin C rich fruits."
        ]
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        
        # Validate input keys
        required_fields = ['Gender', 'Activity_Level', 'Diet_Type', 'Health_Condition', 'Age', 'Weight_kg', 'Height_cm']
        missing_fields = [field for field in required_fields if field not in input_data]
        
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400
        
        # Create DataFrame with proper column names
        input_dict = {
            'Age': input_data['Age'],
            'Weight_kg': input_data['Weight_kg'],
            'Height_cm': input_data['Height_cm'],
            'Gender': input_data['Gender'],
            'Activity_Level': input_data['Activity_Level'],
            'Diet_Type': input_data['Diet_Type'],
            'Health_Condition': input_data['Health_Condition']
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical inputs
        encoded_input = input_df.copy()
        for col in categorical_cols:
            try:
                encoded_input[col] = label_encoders[col].transform([input_df[col].iloc[0]])
            except Exception as e:
                # Handle values not seen during training
                return jsonify({'error': f'Invalid value for {col}: {input_df[col].iloc[0]}'}), 400
        
        # Predict nutrition values
        prediction = model.predict(encoded_input)[0]
        
        # Get food recommendations based on diet type and health condition
        diet_type = input_data['Diet_Type']
        health_condition = input_data['Health_Condition']
        
        # Default recommendations if specific combination not found
        if diet_type not in food_recommendations:
            diet_type = "Omnivore"
        
        if health_condition not in food_recommendations[diet_type]:
            health_condition = "None"
            
        food_rec = random.choice(food_recommendations[diet_type][health_condition])
        
        # Adjust recommendations based on activity level
        activity_level = input_data['Activity_Level']
        activity_note = ""
        
        if activity_level in ["Very Active", "Extremely Active"]:
            activity_note = "Include additional carbohydrates and protein for energy and recovery."
        elif activity_level == "Moderately Active":
            activity_note = "Maintain balanced macronutrients with emphasis on quality protein sources."
        elif activity_level in ["Sedentary", "Lightly Active"]:
            activity_note = "Focus on nutrient-dense foods with controlled portions."
            
        if activity_note:
            food_rec += f" {activity_note}"
        
        return jsonify({
            "Calories": int(prediction[0]),
            "Protein (g)": int(prediction[1]),
            "Carbs (g)": int(prediction[2]),
            "Fats (g)": int(prediction[3]),
            "Recommended Food": food_rec
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)