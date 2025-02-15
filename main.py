import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib


df = pd.read_csv('artistic_emotional_resonance.csv')


X = df[['Color_Dominance', 'Brushstroke_Density', 'Pattern_Complexity', 'Symmetry', 'Artistic_Style']]
y = df['Emotional_Resonance']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")


joblib.dump(model, 'artistic_emotional_resonance_model.pkl')
print("Model saved as 'artistic_emotional_resonance_model.pkl'")

print("\n--- Predict Emotional Resonance of a Painting ---")
user_color_dominance = float(input("Enter Color Dominance (0 to 1): "))
user_brushstroke_density = float(input("Enter Brushstroke Density (0 to 1): "))
user_pattern_complexity = float(input("Enter Pattern Complexity (0 to 1): "))
user_symmetry = float(input("Enter Symmetry (0 to 1): "))
user_artistic_style = int(input("Enter Artistic Style (0: Abstract, 1: Realism, 2: Impressionism): "))

user_input_df = pd.DataFrame({
    'Color_Dominance': [user_color_dominance],
    'Brushstroke_Density': [user_brushstroke_density],
    'Pattern_Complexity': [user_pattern_complexity],
    'Symmetry': [user_symmetry],
    'Artistic_Style': [user_artistic_style]
})


predicted_resonance = model.predict(user_input_df)[0]

print(f"\nPredicted Emotional Resonance: {predicted_resonance:.2f}")
