# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

# Step 2: Create Dataset
data = {
    "Brand": ["Toyota", "BMW", "Toyota", "Ford", "BMW", "Ford"],
    "FuelType": ["Petrol", "Diesel", "Diesel", "Petrol", "Petrol", "Diesel"],
    "Mileage": [1.5, 2.0, 1.8, 1.6, 3.0, 2.2],
    "Price": [15000, 25000, 18000, 16000, 30000, 22000]
}
df = pd.DataFrame(data)

# Step 3: Separate Features and Target
x = df.drop("Price", axis=1) #Drop the column called price and give me the rest drop is a reserved function that will perform the action drop
y = df["Price"]  #the line extracts "Price" column from your Dataframe (df) and stores it in variable called y this is the value your model is trying to predict

# Step 4: One-Hot Encode Categorical Features
categorical_cols = ["Brand", "FuelType"] #these are the columns in my data that contains text("categories")
encoder = OneHotEncoder(sparse_output=False) #it converts  the categories into zeros and ones  sparse out returns regurar numpy array instead of sparse matrix
encoded_cats = encoder.fit_transform(x[categorical_cols])#learn categories (fit) and transforms them to 0/1 format(transform)
# x[categorical_cols] the part of your dataset that has text itemx
#.fit is used to learn patterns or parameters from the data




# Step 5: Combine Encoded + Numeric Data
numerical_cols = x.drop(columns=categorical_cols).values #You are removing the categorical columns (brands, Fueltype) from x so you are left with just the numeric ones the .values just converts the dataframe into a numpy array wich is tensor  likes
x_processed = np.hstack([numerical_cols, encoded_cats]) #this combines the numeric_cols and encoded_cats side by side to form one input array




# Step 6: Split Data
x_train, x_test, y_train, y_test = train_test_split(
    x_processed, y, test_size=0.2, random_state=42
)#this one avoids overfitting and helps measure real world performance
#x-processed All of your input data (numeric + encoded categorical)
#y the target values (what you are predicting -like  car price)
#text-size: use 20% of data for testing 80% for training
#random: sets a random seed so the split is reproducible (same results every time)



# Step 7: Build TensorFlow Model
model = tf.keras.Sequential([ #  Builds a model where each layer feeds directly into the next -one by one in straight line
    tf.keras.layers.Dense(64, activation="relu", input_shape=(x_train.shape[1],)), #this means 64 neurons in the layer activation relu use the Relu function which introduces none linreary x-train.shape[1] tells the model how many input features we have
    tf.keras.layers.Dense(32, activation="relu"), #this has 32 neurons that automatically connects to the previus layer
    tf.keras.layers.Dense(1) #this is the output layer that has 1 neuron because you are predicting 1 number no activation function this is regression not classification
])

"""
 an input layer that expects my training data shape,

a first layer with 64 neurons,

a second layer with 32 neurons,

and an output layer that gives me one number (the price)."   

"""
"""
A neuron in a neural network is a simple math unit that:

Receives input(s) (numbers),

Multiplies them by weights (learned during training),

Adds a bias, and

Applies an activation function (like ReLU or sigmoid) to produce an output.

"""



# Step 8: Compile and Train Model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(x_train, y_train, epochs=100, validation_split=0.1, verbose=0)

# Step 9: Evaluate Model
loss, mae = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Mean Absolute Error: ${mae:.2f}")

# Step 10: Predict
predictions = model.predict(x_test)
for i, pred in enumerate(predictions):
    print(f"Predicted: ${int(pred[0])}, Actual: ${y_test.iloc[i]}")
