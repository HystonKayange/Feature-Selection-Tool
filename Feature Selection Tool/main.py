import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import tkinter as tk
from tkinter import filedialog
from feature_selection import FeatureSelector
from dataset import GenericDataset








def open_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    welcome_message = "Welcome to a Feature Selection tool for Machine Learning!\n\nSelect a dataset to get started."
    title = "FeatureSelectorTool"
    tk.messagebox.showinfo(title, welcome_message)

    file_path = filedialog.askopenfilename(
        title="Select a dataset file", filetypes=[("CSV files", "*.csv"), ("TXT files", "*.txt"), ("All files", "*.")]
    )
    if not file_path:
        print("No file selected. Exiting the program.")
        exit()
    return file_path

def get_dataset(file_path):
    dataset = GenericDataset(file_path)
    # Convert feature names to Strings
    dataset.features.columns = dataset.features.columns.astype(str)
    return dataset
   

def display_num_features(features):
    
    print(f"The dataset has {features} features.")
    

def perform_task(dataset, dataset_name):
    print(f"Choose a task for the '{dataset_name}' dataset:")

    print("1. Classification Task")
    print("2. Regression Task")
    print("3. Exit")

    task_choice = input("Enter the number of your choice: ")

    if task_choice == '1':
        task_type = 'classification'
    elif task_choice == '2':
        task_type = 'regression'
    elif task_choice == '3':
        print("Exiting the program.")
        exit()
    else:
        print("Invalid choice. Exiting.")
        exit()
    
    

    # Perform the selected task (classification or regression) using the dataset
    if task_type == 'classification':
        print(f"Performing Classification Task with Feature Selection on the '{dataset_name}' dataset")

        # Ask the user for the number of features they want
        # Ask the user for the number of features they want
        num_features_str = input("Enter the number of features you want to select: ")
        try:
            num_features = int(num_features_str)
            if num_features <= 0 or num_features > dataset.features.shape[1]:
                print("Invalid number of features. Exiting.")
                exit()
        except ValueError:
            print("Invalid input. Please enter a valid number. Exiting.")
            exit()
        top_k = num_features
        feature_selector = FeatureSelector(method='filter', k=num_features, top_k=top_k)

        # Assuming 'features' contains your input features and 'label' contains your target variable
        X_train = dataset.features
        y_train = dataset.labels

        # Perform feature selection
        X_train_selected = feature_selector.fit_transform(X_train, y_train)

        # Print the selected features (optional)
        print("Selected Features:")
        print(X_train_selected)

        # Train a model to get feature importances
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X_train_selected, y_train)

        # Get feature importance 
        feature_importances = model.feature_importances_

        # plot feature importance
        feature_selector.plot_feature_importance(X_train.columns, feature_importances)

        

        # Visualize distribution plots
        feature_selector.plot_feature_distribution(X_train_selected)
        from sklearn.preprocessing import StandardScaler
         # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_train_selected, y_train, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

       

        # Train a Logistic Regression model
        classification_model = LogisticRegression(max_iter=1000)
        classification_model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = classification_model.predict(X_test_scaled)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on the test set: {accuracy}")

    elif task_type == 'regression':
        print(f"Performing Regression Task on the '{dataset_name}' dataset")

        # Ask the user for the number of features they want (optional for regression)
        num_features = int(input("Enter the number of features you want to select: "))
        feature_selector = FeatureSelector(method='filter', k=num_features)

        # Assuming 'features' contains your input features and 'label' contains your target variable
        X_train = dataset.features
        y_train = dataset.labels

        # Perform feature selection (if specified)
        X_train_selected = feature_selector.fit_transform(X_train, y_train)
        print("Selected Features:")
        print(X_train_selected)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_train_selected, y_train, test_size=0.2, random_state=42)

        # Train a Linear Regression model
        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = regression_model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error on the test set: {mse}")

    else:
        print("Invalid task type. Exiting.")
        exit()

if __name__ == "__main__":
    file_path = open_file_dialog()
    dataset = get_dataset(file_path)
    border = "___________________________________________________________________________________________"
    print(border)
    print("Dataset.head()")
    print(border)
    print(dataset.features.head())
    print(border)
    display_num_features(dataset.features.shape[1])
    print(border)

    # Ask the user whether they want to perform a classification or regression task
    perform_task(dataset, dataset_name=file_path)
