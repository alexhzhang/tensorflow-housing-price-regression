# **California Housing Price Prediction — TensorFlow Linear Regression**

This project implements an end-to-end machine learning workflow using **TensorFlow/Keras** to predict housing prices from the 1990 California census dataset. The work includes data cleaning, exploratory data analysis, feature engineering, model training, and performance evaluation.

## 🚀 **Project Summary**

* Built a **linear regression model** using TensorFlow/Keras
* Loaded and analyzed the California Housing Dataset (population, rooms, income, etc.)
* Implemented **EDA**: scatterplot matrix, correlation analysis, statistical summary
* Trained models using multiple features, hyperparameter tuning, and synthetic feature creation
* Evaluated predictions with RMSE and visualized performance (loss curves, regression fits)

## 🧠 **Key Concepts Demonstrated**

* Data cleaning + preprocessing (scaling, anomaly identification)
* Correlation analysis and exploratory data visualization
* TensorFlow model construction (`Dense` layer with RMSProp optimizer)
* Batch training, epochs, and root-mean-squared-error tracking
* Synthetic feature engineering (`rooms_per_person`)
* Evaluating predictive power vs. individual features

## 📁 **File Structure**

```
linear_regression_california_housing.ipynb   # Jupyter notebook (full workflow)
README.md                                     # Project documentation
```

## 📊 **Results**

* Baseline features (`total_rooms`, `population`) showed limited predictive strength
* Synthetic feature (`rooms_per_person`) improved performance
* Strongest correlation identified: **median_income** → house value
* Final model demonstrates full TensorFlow regression workflow on a real dataset

## 📦 **Dataset**

California Housing Dataset (public):
[https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv](https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv)

## 🔧 **Tech Stack**

* **Python**
* **TensorFlow / Keras**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**
