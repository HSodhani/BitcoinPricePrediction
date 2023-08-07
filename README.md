# Bitcoin Price Prediction using Linear Regression, Polynomial Regression, and Support Vector Machine

In this project, we use three different machine learning algorithms - Linear Regression, Polynomial Regression, and Support Vector Machine - to predict the price of Bitcoin. The aim of this project is to compare the performance of the different models and identify the best algorithm for predicting Bitcoin prices.

# Dataset
The data used in this project is taken from [kaggle](https://www.kaggle.com/datasets/hardiksodhani/bitcoinprice "kaggle"). It consists of the per minute price of Bitcoin from 2017 to March 3, 2022. We will use this data to train our models and make predictions on future Bitcoin prices.

# Prerequisites
In order to run this project, you will need to have the following software installed:

- Python 3.6+
- Jupyter Notebooks
- Scikit-learn
- Pandas
- Numpy
- Matplotlib

# To install the required packages, you can use pip:

`pip install scikit-learn pandas numpy matplotlib`

# Usage
To run the project, simply open the Jupyter Notebook `BitcoinPricePrediction.ipynb`. This notebook contains all the code required to train the models and make predictions on future Bitcoin prices. You can run each cell in the notebook to see the output and results.

# Results
After training and testing the models, we found that the Support Vector Machine algorithm performed the best in predicting Bitcoin prices. The performance of each algorithm is summarized below:

- Linear Regression: Mean Squared Error (MSE) = 119270830, R-squared (R2) score = 0.61
- Polynomial Regression: Mean Squared Error (MSE) = 60187071, R-squared (R2) score = 0.80
- Support Vector Machine: Mean Squared Error (MSE) = 7115225, R-squared (R2) score = 0.91

# Conclusion
In conclusion, we have demonstrated how to use three different machine learning algorithms to predict the price of Bitcoin. We found that the Support Vector Machine algorithm outperformed both Linear Regression and Polynomial Regression in terms of accuracy. However, it is important to note that the performance of these models may vary depending on the dataset and time period used. This project serves as a starting point for further exploration into Bitcoin price prediction using machine learning.

# Contributors
- [Rucha Chotalia](https://github.com/Ruchachotalia "Rucha Chotalia")
