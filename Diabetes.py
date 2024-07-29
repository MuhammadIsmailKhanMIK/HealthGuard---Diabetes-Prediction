import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import messagebox, font

# Load Pima Indians Diabetes Dataset from a local file
file_path = "diabetes.csv"
data = pd.read_csv(file_path)

# Separate features (X) and target (y) from the dataset
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the test set results
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)

# GUI for User Input and Prediction
class HealthGuardApp:
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.title("HealthGuard: Diabetes Prediction")
        self.root.geometry("500x800")
        self.root.config(bg="#34495e")
        
        # Define fonts for the UI elements
        self.header_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=12)
        self.entry_font = font.Font(family="Helvetica", size=12)
        self.button_font = font.Font(family="Helvetica", size=14, weight="bold")
        self.result_font = font.Font(family="Helvetica", size=14, weight="bold")

        self.create_widgets()

    def create_widgets(self):
        # Dictionary to hold the label and entry widgets
        self.labels_entries = {}
        self.columns = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]

        # Corresponding questions for the columns
        questions = [
            "Number of Pregnancies:",
            "Plasma Glucose Concentration:",
            "Diastolic Blood Pressure (mm Hg):",
            "Triceps Skinfold Thickness (mm):",
            "2-hour Serum Insulin (mu U/ml):",
            "Body Mass Index (BMI):",
            "Diabetes Pedigree Function:",
            "Age:"
        ]

        # Header label for the UI
        header_label = tk.Label(self.root, text="Diabetes Prediction", bg="#34495e", fg="white", font=self.header_font)
        header_label.pack(pady=20)

        # Frame to hold the form elements
        form_frame = tk.Frame(self.root, bg="#34495e")
        form_frame.pack(pady=10)

        # Create label and entry widgets for each question
        for question, column in zip(questions, self.columns):
            frame = tk.Frame(form_frame, bg="#34495e")
            frame.pack(pady=5, fill="x")

            label = tk.Label(frame, text=question, bg="#34495e", fg="white", font=self.label_font, anchor="w", width=30)
            label.pack(side=tk.LEFT, padx=10)
            
            entry = tk.Entry(frame, font=self.entry_font)
            entry.pack(side=tk.RIGHT, padx=10, fill="x", expand=True)
            
            self.labels_entries[column] = entry
        
        # Predict button to trigger the prediction
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_disease, bg="#2980b9", fg="white", font=self.button_font)
        self.predict_button.pack(pady=30)
        
        # Label to display the prediction result
        self.result_label = tk.Label(self.root, text="", bg="#34495e", fg="white", font=self.label_font)
        self.result_label.pack(pady=10)

    def predict_disease(self):
        try:
            # Collect the input features from the user
            features = [float(self.labels_entries[column].get()) for column in self.columns]
            input_data = pd.DataFrame([features], columns=self.columns)
            
            # Standardize the input features
            input_data = scaler.transform(input_data)
            
            # Make the prediction
            prediction = model.predict(input_data)[0]
            
            # Display the result with enhanced styling
            result_text = f"Predicted Disease: {'Positive' if prediction == 1 else 'Negative'}"
            self.result_label.config(text=result_text, font=self.result_font)
        except ValueError:
            # Show an error message if the input data is invalid
            messagebox.showerror("Input Error", "Please enter valid data")

if __name__ == "__main__":
    # Create the main window and start the application
    root = tk.Tk()
    app = HealthGuardApp(root)
    root.mainloop()