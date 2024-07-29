import pandas as pd

# URL of the dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# Load the dataset
data = pd.read_csv(url)

# Save to a local CSV file
data.to_csv("diabetes.csv", index=False)

print("Dataset downloaded and saved as diabetes.csv")