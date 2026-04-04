import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Loading data from CSV...")

# --- 1. LOAD THE DATA ---
X = [] # This will hold the coordinates (the math)
y = [] # This will hold the labels (the letters 'A', 'B', 'C')

with open('hand_data.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row
    for row in reader:
        y.append(row[0]) # The first column is the letter
        X.append([float(val) for val in row[1:]]) # The rest are the coordinates

# Convert lists to NumPy arrays for machine learning
X = np.array(X)
y = np.array(y)

print(f"Total examples loaded: {len(y)}")

# --- 2. SPLIT THE DATA ---
# We keep 20% of the data hidden from the AI to test it later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAIN THE MODEL ---
print("Training the AI model... (this might take a few seconds)")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# --- 4. TEST THE MODEL ---
# Let's see how well it learned by testing it on the hidden 20%
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(f"Training Complete! Model Accuracy: {score * 100:.2f}%")

# --- 5. SAVE THE "BRAIN" ---
# We save the trained model as a .pkl file so our live app can use it
with open('sign_language_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Model successfully saved as 'sign_language_model.pkl'")