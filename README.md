# Heat Classification Project using Neural Networks

## Overview

This project aims to develop a feed-forward neural network model for classifying the presence or absence of heart disease based on the Cleveland Heart Disease dataset. The dataset consists of 303 rows, each representing information about an individual patient, with 14 attribute columns and a binary class column indicating the presence or absence of heart disease.

## Dataset

- **Name:** Cleveland Heart Disease dataset
- **Source:** [Link](https://your_dataset_link_here)
- **Structure:** CSV file with 303 rows and 14 columns
- **Attributes:** Detailed information about patients
- **Target:** Binary class indicating the presence of heart disease

## Preprocessing

1. **Loading the dataset:**
   - Pandas DataFrame with column names.
   - Replacing "?" with NaN.
   - Removing rows with NaN values.

2. **Splitting the dataset:**
   - Three subsets: training (70%), validation (20%), and test (10%).
   - Random seed for reproducibility.

## Neural Network Model

### Architecture

- **Model Type:** Sequential Neural Network
- **Layers:**
  1. Input Layer:
     - Type: Dense (Fully Connected)
     - Units: 256
     - Activation: ReLU
  2. Hidden Layer:
     - Type: Dense (Fully Connected)
     - Units: 128
     - Activation: ReLU
  3. Hidden Layer:
     - Type: Dense (Fully Connected)
     - Units: 32
     - Activation: ReLU
  4. Output Layer:
     - Type: Dense (Fully Connected)
     - Units: 5 (for multi-class classification)
     - Activation: Softmax

### Compilation

- **Optimizer:** Adam with a learning rate of 0.0003.
- **Loss Function:** Categorical Crossentropy (for multi-class classification).
- **Metrics:** Accuracy (used for evaluation during training).

## Model Implementation

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def create_model():
    model = Sequential()
    model.add(Dense(256, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, activation='softmax'))
    
    adam = Adam(lr=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

def print_model_summary(model):
    print(model.summary())

# Create the model
model = create_model()

# Print model summary
print_model_summary(model)
```

## Model Training

1. Plot the completed model for visualization.
2. Use the `fit()` function to train the model on the training dataset.

## Model Evaluation

```python
from sklearn.metrics import classification_report, accuracy_score

# Predictions on the test set
predictions = model.predict(X_test)

# Convert one-hot encoded predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Compute accuracy score
accuracy = accuracy_score(y_test, predicted_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Create classification report
report = classification_report(y_test, predicted_labels)
print('Classification Report:\n', report)
```

## Conclusion

By using Neural Networks we are able to effectively predict the presence or absence of heart disease based on patient attributes.
