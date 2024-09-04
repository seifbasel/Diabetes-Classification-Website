# Diabetes-Classification

# Run
To run the project run the application.py file

# Project Summary
The goal of this project is to develop a classification model for diabetes prediction. The project utilizes various machine learning algorithms and techniques to analyze a dataset of patients' health information and determine the likelihood of an individual having diabetes. 

## Dataset Analysis
The dataset contains essential features such as glucose rate, weight, age, blood pressure, cholesterol level, and gender. In-depth analysis of the dataset reveals interesting insights related to diabetes and its risk factors.

## Model Development
Several classification models were explored, including K-nearest neighbors (KNN), Support Vector Machines (SVM), and a Neural Network using MLPClassifier. The KNN model achieved the highest accuracy of 91.76% with an optimal k value of 5. The SVM model yielded an accuracy of 92.94%, demonstrating its effectiveness in diabetes classification. However, the neural network model's accuracy was lower at 83.3% due to the small dataset size, rendering it unsuitable for practical use.

## Web Application
To provide an interactive user experience, a web application was developed using Flask, HTML, and Bootstrap. The application allows users to input their health information and obtain a prediction regarding the presence or absence of diabetes based on the selected classification model.

## Conclusion
This project highlights the significance of classification models in diabetes prediction. The accuracy achieved by the KNN and SVM models emphasizes their potential for accurate classification. The web application serves as a practical tool for individuals to assess their risk of diabetes and seek appropriate medical guidance. Further enhancements and model refinements can be explored to improve prediction accuracy and expand the scope of the project.

## Demo

[Download the demo video](projectdemo.mp4)

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Diabetes-Classification.git

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Diabetes-Classification.git
   
2. **Clone the repository:**

   ```bash
   cd Diabetes-Classification
   
3. **Clone the repository:**

   ```bash
   pip install -r requirements.txt
   
4. **Clone the repository:**

   ```bash
   python application.py
   