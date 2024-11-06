# Feature Selector tool 

## Overview
This Python tool is designed for automatic feature selection for machine learning datasets . It provides a seamless way to handle missing values, detect and encode categorical variables, and offers insights into feature importance,feature corrilletion and  distribution.It is designed for both classification and regression tasks.

## Usage
     1. Navigate to the project directory [cd feature_selection_tool]
     2. Run the tool [ python main.py]
     3. Choose a dataset using the file explorer when prompted
     4. Follow the on-screen instructions to perfom feature selection

## Features

       1. Automatic Handling of Missing Values: The tool intelligently manages missing values for a seamless analysis experience.
       2. Automatic Detection and Encoding of Categorical Variables: It identifies categorical variables and applies appropriate encoding for accurate feature selection
       3. Insights into Feature Importance and Distribution: Gain a deeper understanding of your dataset by exploring feature distributions.
       4. It accepts csv and txt files.

## Visualizations
The Feature Selector Tool includes visuallization methods to inspect characteristics of a dataset
### Feature Importance
![Feature Importance](images/Figure_1.png)

### Feature Distribution
![Feature Importance](images/Figure_3.png)

### Prerequisites

- Python (>=3.7)
- pandas==2.1.2
- numpy==1.26.1
- scikit-learn==1.3.2
- feature-engine==1.6.2
- torch==2.1.0
- matplotlib==3.8.1
- seaborn==0.13.0


## Limitations

**Large Datasets:**

- The tool is designed to handle datasets of various sizes, but users may encounter challenges with very large datasets. Issues related to memory usage and computation time could arise. It's recommended to assess the computational resources available before applying feature selection to large datasets.

**Outliers:**

- While the tool provides automated handling of missing values and categorical variables, it may not robustly handle outliers. Users are advised to preprocess their data to address outliers before initiating the feature selection process.

**Dataset Characteristics:**

- The performance of the tool may vary depending on the characteristics of the dataset. Users should be aware that not all datasets are guaranteed to be fully processed, and some datasets may require additional preprocessing steps to align with the tool's capabilities.

**Recommendations:**

- Consider subsampling or preprocessing large datasets before using the tool to manage memory and computation challenges.
- Prioritize outlier detection and handling as part of your data preprocessing workflow.
- Be attentive to the characteristics of your dataset, and explore additional preprocessing steps if needed.

**Future Improvements:**

Our team is committed to refining and enhancing the tool. We plan to address these limitations in future updates to provide a more robust and versatile feature selection experience.

Feel free to check our [Contributing Guidelines](CONTRIBUTING.md) for any additional information on reporting issues or contributing to the tool's development.

## Datasets Tested

Our feature selection tool has been tested on the following datasets:

1. **Heart Disease Dataset**
   - Description: This dataset contains information about patients and features related to heart disease.
   - Source: [Link to Dataset](provide_link_here)

2. **Diabetes Dataset**
   - Description: A dataset with features related to diabetes and health indicators.
   - Source: [Link to Dataset](provide_link_here)

3. **Health Stroke Dataset**
   - Description: Dataset focusing on features related to strokes and healthcare information.
   - Source: [Link to Dataset](provide_link_here)

4. **MovieLens Dataset**
   - Description: A movie recommendation dataset with user and movie features.
   - Source: [Link to Dataset](provide_link_here)

Feel free to explore and test our feature selection tool on these datasets to see how it performs in different scenarios. If you have additional datasets you'd like to test or share, please let us know!
