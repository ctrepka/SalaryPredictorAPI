# SalaryPredictorAPI

This is a simple sample API using Flask in Python to expose an ML Model. It is not intended for production purposes, but is a simple exercise to exhibit the ability to easily generate endpoints in flask that expose ML Models for predictions.

In this example, we use a very small (too small to be reliable) dataset of salary data. The columns are a categorical column of professions, years of experience, and salary. Upon starting the Flask app, the application ingests a Salary_Data.csv, trains an ML model using a Multiple Regression Analysis, and then dumps that model to a bytestream using Pickle. The bytestream is then exposed on the Flask app, which runs predictions via a POST request to the endpoint '/predict'.
