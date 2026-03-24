import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
with open('Kdata.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Extract data and labels
data = data_dict['data']
labels = np.array(data_dict['labels'])

print(data[0])
# Determine the maximum feature length
max_length = max(len(sample) for sample in data)
print(max_length)

for i, sample in enumerate(data):                         #extra_1 (for printing dataset with 84-coords)
    if len(sample) != 42:
        print(f"Sample {i} has {len(sample)} features!")  # Debugging

# Pad feature vectors to the same length
def pad_features(sample, length):
     return sample + [0] * (length - len(sample))  # Fill missing values with zeros

data = [sample[:42] for sample in data]  # Keep only first 42 coordinates         extr_3
data_padded = np.array(data, dtype=np.float32)# Convert to NumPy array
print("Final shape of data_padded:", data_padded.shape)  # Should be (num_samples, 42)

#data_padded = np.array([pad_features(sample, max_length) for sample in data], dtype=np.float32)           (ORIGNAL)

# Debugging: Check the shape of the feature matrix                  extr_2
print("Shape of data_padded:", data_padded.shape)  # Should be (num_samples, num_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, random_state=42, stratify=labels)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# print(f"Model training complete! Accuracy: {accuracy:.4f}")

# Save trained model
with open('Kmodel.pickle', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Trained model saved as Kmodel.pickle")