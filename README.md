# 🚗 Car Price Prediction Using TensorFlow and One-Hot Encoding

This project uses **TensorFlow**, **Pandas**, and **Scikit-learn** to predict car prices based on features like brand, fuel type, and mileage.

It includes handling of **categorical data** using `OneHotEncoder` and building a simple neural network using **Keras (TensorFlow backend)**.

---

## 📊 Dataset

A small sample dataset is hardcoded with the following columns:

- `Brand`: Categorical (e.g., Toyota, BMW, Ford)
- `FuelType`: Categorical (e.g., Petrol, Diesel)
- `Mileage`: Numeric
- `Price`: Target variable (what we want to predict)

---

## 🧠 What This Project Covers

- Data preparation using **Pandas**
- Handling **categorical variables** with OneHotEncoding
- Combining **numerical** and **encoded categorical** features
- Splitting data into **training** and **testing sets**
- Building a **neural network** using `tf.keras`
- Evaluating model performance
- Making predictions with the trained model

---

## ⚙️ How It Works

1. **Preprocess the Data**
   - Split features (`x`) and target (`y`)
   - Use `OneHotEncoder` to encode `Brand` and `FuelType`
   - Combine encoded features with `Mileage` (numerical)

2. **Split the Dataset**
   - 80% for training
   - 20% for testing

3. **Build the Model**
   - Input → Dense(64, relu)
   - Hidden → Dense(32, relu)
   - Output → Dense(1)

4. **Train & Evaluate**
   - Optimizer: Adam
   - Loss: Mean Squared Error (MSE)
   - Metric: Mean Absolute Error (MAE)

---

## 🔧 Technologies Used

- Python 🐍
- TensorFlow / Keras 🧠
- Pandas 📊
- NumPy
- Scikit-learn 🔬

---

## 🏁 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
