 Below is a README.md file for Exploratory Data Analysis and Modelling for Loan Application Prediction :

# Exploratory Data Analysis and Modelling for Loan Application Prediction

## Overview
This project performs exploratory data analysis and applies machine learning models to predict the success of loan applications.
The goal is to build models that can accurately predict loan approval status to help estimate approval chances even before formally applying for a loan.
The dataset used contains details of loan applicants, such as gender, income, credit history, etc. 

Two models are implemented - Support Vector Machines (SVM) and Logistic Regression. SVM is chosen as the primary algorithm and Logistic Regression as the secondary algorithm.

## Data
The loan application dataset used contains details of 614 loan applicants with the following features:

- Categorical features: Gender, Married, Dependents, Education, Self_Employed, Property_Area  
- Numerical features: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
- Target variable: Loan_Status (Loan approved or not)

## Data Analysis
The data analysis steps include:

- Importing libraries like NumPy, Pandas, Matplotlib etc. for data manipulation and visualization
- Importing dataset and exploring it to understand shape, columns, data types etc. 
- Handling missing values 
- Analyzing distribution of single attributes like gender and loan term using count plots
- Finding relationships between attributes using crosstabs and correlation analysis 
- Identifying and removing redundant attributes
- Spotting outliers and skewed distributions from visualizations
- Applying transformations to handle skewness and scaling data

## Methods
The following steps are performed:

- **Data Cleaning**: Handle missing values, remove unnecessary features, encode categorical variables
- **Exploratory Data Analysis**: Visualize distributions of key variables, analyze correlations, identify outliers
- **Preprocessing**: Normalize data, handle class imbalance with SMOTE 
- **Modeling**: Implement SVM (primary algorithm) and Logistic Regression (secondary algorithm)
- **Evaluation**: Compare model accuracy, plot confusion matrix and ROC curve

## Model Implementation
The main steps involved:

- Encoding categorical data using OneHotEncoder
- Balancing dataset using SMOTE 
- Splitting data into train and test sets
- Building SVM and Logistic Regression models
- Evaluating model accuracy using classification report, confusion matrix etc.

## Results
- The SVM model achieved an accuracy of 70.21% 
- The Logistic Regression model achieved an accuracy of 68.08%

## Key Findings

- Both SVM and Logistic Regression achieve decent accuracy, with SVM performing slightly better at 70.2% vs 68.1% for Logistic Regression
- Applicants with longer loan repayment term and no credit history tend to get rejected more often 
- Higher applicant income and lower loan amount generally increase chances of approval
- Semiurban property area has higher approval rate compared to rural/urban areas

Overall, the SVM performs better for this dataset. The models can be further tuned and optimized.

## Usage
The Jupyter notebook contains detailed documentation and code. 
To use:
1. Clone this repo
2. Install dependencies like Numpy, Pandas, Scikit-learn etc.  
3. Run the notebook cell by cell

Let me know if any clarification is needed on this README or project!
