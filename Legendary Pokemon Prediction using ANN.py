#Perform data preprocessing and prepare the data for feeding into the neural network.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Pokemon Data.csv')

df.columns = df.columns.str.strip()

df.head()
df.info()
df.describe()

df = df.drop(['#', 'Name', 'Type 1', 'Type 2','Generation'], axis=1)

df.isnull().sum()
df.duplicated().sum()

X = df.drop('Legendary', axis=1)
y = df['Legendary']
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(# type: ignore
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)# type: ignore
X_test = scaler.transform(X_test)# type: ignore
X_test = scaler.transform(X_test)# type: ignore
np.array(X_train)
np.array(X_test) # type: ignore
np.array(y_train) # type: ignore
np.array(y_test) # type: ignore

#Build the neural network model using Keras and train it on the preprocessed data.
from keras.models import Sequential# type: ignore
from keras.layers import Dense# type: ignore
from keras.optimizers import Adam# type: ignore
from keras.callbacks import EarlyStopping# type: ignore
from sklearn.metrics import classification_report, confusion_matrix# type: ignore

# Build the neural network model
model = Sequential()
# Add input layer and first hidden layer
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))# type: ignore
# Add second hidden layer
model.add(Dense(units=32, activation='relu'))# type: ignore
# Add output layer
model.add(Dense(units=1, activation='sigmoid'))# type: ignore
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])# type: ignore
# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)# type: ignore
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)# type: ignore
# Print the test accuracy
print(f'Test Accuracy: {test_accuracy:.4f}')
#use the trained model to make predictions on the test set and evaluate its performance using classification metrics.
# Make predictions on the test set
y_pred = model.predict(X_test)# type: ignore
# Convert probabilities to binary predictions
y_pred_binary = (y_pred > 0.5).astype(int)# type: ignore
# Generate classification report
report = classification_report(y_test, y_pred_binary)# type: ignore
print(report)# type: ignore
# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)# type: ignore
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))# type: ignore
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Legendary', 'Legendary'], yticklabels=['Not Legendary', 'Legendary'])# type: ignore
plt.xlabel('Predicted')# type: ignore
plt.ylabel('Actual')# type: ignore
plt.title('Confusion Matrix')# type: ignore
plt.show()# type: ignore
# Plot training and validation loss over epochs
plt.figure(figsize=(10, 6))# type: ignore
plt.plot(history.history['loss'], label='Training Loss')# type: ignore
plt.plot(history.history['val_loss'], label='Validation Loss')# type: ignore
plt.title('Training and Validation Loss')# type: ignore
plt.xlabel('Epochs')# type: ignore
plt.ylabel('Loss')# type: ignore
plt.legend()# type: ignore
plt.show()# type: ignore