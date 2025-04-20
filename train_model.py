import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# Check if dataset exists, if not, create a synthetic one
try:
    df = pd.read_csv('nutrition_recommendation_dataset.csv')
    print("Dataset found, loading...")
except FileNotFoundError:
    print("Dataset not found, creating synthetic dataset...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Define possible values for categorical features
    genders = ['Male', 'Female']
    activity_levels = ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active', 'Extremely Active']
    diet_types = ['Omnivore', 'Vegetarian', 'Vegan', 'Pescatarian', 'Keto', 'Paleo']
    health_conditions = ['None', 'Diabetes', 'Hypertension', 'Anemia']
    
    # Sample random values for each feature
    ages = np.random.randint(18, 80, n_samples)
    weights = np.random.uniform(45, 120, n_samples)
    heights = np.random.uniform(150, 200, n_samples)
    genders_sampled = np.random.choice(genders, n_samples)
    activity_levels_sampled = np.random.choice(activity_levels, n_samples)
    diet_types_sampled = np.random.choice(diet_types, n_samples)
    health_conditions_sampled = np.random.choice(health_conditions, n_samples)
    
    # Create synthetic target values based on features
    # For simplicity, we'll use a combination of features to generate targets
    
    # Base calorie calculations (BMR using Mifflin-St Jeor Equation)
    calories = np.zeros(n_samples)
    proteins = np.zeros(n_samples)
    carbs = np.zeros(n_samples)
    fats = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Calculate BMR
        if genders_sampled[i] == 'Male':
            bmr = 10 * weights[i] + 6.25 * heights[i] - 5 * ages[i] + 5
        else:
            bmr = 10 * weights[i] + 6.25 * heights[i] - 5 * ages[i] - 161
        
        # Apply activity multiplier
        activity_multipliers = {
            'Sedentary': 1.2,
            'Lightly Active': 1.375,
            'Moderately Active': 1.55,
            'Very Active': 1.725,
            'Extremely Active': 1.9
        }
        
        calories[i] = bmr * activity_multipliers[activity_levels_sampled[i]]
        
        # Adjust for diet type
        diet_adjustments = {
            'Omnivore': 0,
            'Vegetarian': -50,
            'Vegan': -100,
            'Pescatarian': -30,
            'Keto': 50,
            'Paleo': 30
        }
        
        calories[i] += diet_adjustments[diet_types_sampled[i]]
        
        # Adjust for health conditions
        condition_adjustments = {
            'None': 0,
            'Diabetes': -150,
            'Hypertension': -100,
            'Anemia': 50
        }
        
        calories[i] += condition_adjustments[health_conditions_sampled[i]]
        
        # Add some noise to make it realistic
        calories[i] += np.random.normal(0, 50)
        
        # Calculate macronutrients based on diet type
        if diet_types_sampled[i] == 'Keto':
            proteins[i] = weights[i] * 1.6  # Higher protein for keto
            fats[i] = (calories[i] * 0.75) / 9  # 75% calories from fat
            carbs[i] = (calories[i] * 0.05) / 4  # 5% calories from carbs
        elif diet_types_sampled[i] == 'Vegan' or diet_types_sampled[i] == 'Vegetarian':
            proteins[i] = weights[i] * 1.3  # Slightly higher for plant-based
            fats[i] = (calories[i] * 0.25) / 9  # 25% calories from fat
            carbs[i] = (calories[i] * 0.55) / 4  # 55% calories from carbs
        else:
            proteins[i] = weights[i] * 1.5  # General recommendation
            fats[i] = (calories[i] * 0.3) / 9  # 30% calories from fat
            carbs[i] = (calories[i] * 0.45) / 4  # 45% calories from carbs
    
    # Create the DataFrame
    data = {
        'Age': ages,
        'Weight_kg': weights,
        'Height_cm': heights,
        'Gender': genders_sampled,
        'Activity_Level': activity_levels_sampled,
        'Diet_Type': diet_types_sampled,
        'Health_Condition': health_conditions_sampled,
        'Calories_Needed': np.round(calories).astype(int),
        'Protein_g': np.round(proteins).astype(int),
        'Carbs_g': np.round(carbs).astype(int),
        'Fats_g': np.round(fats).astype(int)
    }
    
    df = pd.DataFrame(data)
    
    # Add food recommendations (just placeholder text for synthetic data)
    food_recommendations = []
    for i in range(n_samples):
        if diet_types_sampled[i] == 'Omnivore':
            food_recommendations.append("Balanced diet with lean protein, whole grains, fruits, and vegetables")
        elif diet_types_sampled[i] == 'Vegetarian':
            food_recommendations.append("Plant-based proteins, dairy, eggs, whole grains, and plenty of vegetables")
        elif diet_types_sampled[i] == 'Vegan':
            food_recommendations.append("Plant proteins like beans, lentils, tofu, whole grains, nuts, seeds, and vegetables")
        elif diet_types_sampled[i] == 'Pescatarian':
            food_recommendations.append("Fish, seafood, plant proteins, whole grains, fruits, and vegetables")
        elif diet_types_sampled[i] == 'Keto':
            food_recommendations.append("High-fat foods, moderate protein, low-carb vegetables")
        elif diet_types_sampled[i] == 'Paleo':
            food_recommendations.append("Lean meats, fish, fruits, vegetables, nuts, and seeds")
    
    df['Recommended_Foods'] = food_recommendations
    
    # Save the synthetic dataset
    df.to_csv('nutrition_recommendation_dataset.csv', index=False)
    print("Synthetic dataset created and saved.")

# Encode categorical columns
label_encoders = {}
categorical_cols = ['Gender', 'Activity_Level', 'Diet_Type', 'Health_Condition']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Targets
X = df[['Age', 'Weight_kg', 'Height_cm', 'Gender', 'Activity_Level', 'Diet_Type', 'Health_Condition']]
y = df[['Calories_Needed', 'Protein_g', 'Carbs_g', 'Fats_g']]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'model.pkl')
joblib.dump(label_encoders, 'encoders.pkl')

# Make sure the processed data includes all columns
df.to_csv('processed_data.csv', index=False)
print("Model training complete.")