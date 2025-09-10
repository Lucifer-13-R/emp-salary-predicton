# Employee Salary Prediction App

This project develops a machine learning application to predict an individual's income level (<=50K or >50K) based on various demographic and employment-related features. The application is built using Python, Scikit-learn for the machine learning model, and Streamlit for the interactive web interface. Optionally, it integrates the Google Gemini API for additional insights.

## Features

*   **Data Preprocessing:** Handles missing values and converts categorical features into a format suitable for machine learning.
*   **Logistic Regression Model:** A simple yet effective classification model for binary income prediction.
*   **Interactive Web Interface:** Built with Streamlit, allowing users to input employee data and get instant predictions.
*   **Model Persistence:** The trained model is saved and loaded, avoiding retraining every time the app runs.
*   **Gemini API Integration (Optional):** Provides contextual information or explanations related to the dataset and predictions.

## Dataset

The project uses the `adult 3.csv` dataset, which contains information extracted from the 1994 Census database.

**Source:** `MultipleFiles/adult 3.csv`

## Project Structure
