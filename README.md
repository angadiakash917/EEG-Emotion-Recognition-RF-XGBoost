EEG-Based Emotion Recognition Using Random Forest and XGBoost

This project implements a full EEG-based emotion recognition system using classical ensemble machine learning techniques. The goal is to classify emotional states—Negative, Neutral, Positive—using 510 selected EEG statistical, spectral, wavelet, covariance, and correlation-based features. The dataset consists of 2132 samples and originally 2549 extracted features, which were reduced for improved model efficiency and reliability. The system achieves high predictive performance using Random Forest and XGBoost models, making it suitable for Pattern Recognition coursework and research applications.

Key Features

EEG emotion classification using machine learning

Two state-of-the-art ensemble models:

Random Forest: 98.59% accuracy

XGBoost: 99.22% accuracy

Handles any EEG feature CSV file with the same structure

Auto-generates:

Confusion matrices

Feature importance graphs

Accuracy comparison

Fully reproducible Python pipeline

No manual intervention required

Project Structure
├── pr.py             
├── dataset/          
├── results/          
├── README.md         
└── requirements.txt  

Dataset Information

2132 total samples

2549 raw features, reduced to 510 selected features

Feature categories include:

FFT spectral power

Time-domain statistics

Wavelet energy features

Covariance matrices

Correlation metrics

Emotion labels:

0 = Negative

1 = Neutral

2 = Positive

Installation

Clone the repository:

git clone https://github.com/yourusername/EEG-Emotion-Recognition.git
cd EEG-Emotion-Recognition


Install dependencies:

pip install -r requirements.txt

How to Run

Execute the main script:

python pr.py


The script automatically:

Loads and encodes the dataset

Selects 510 features

Trains Random Forest and XGBoost

Generates confusion matrices

Saves feature importance plots

Prints accuracy and model comparison

Results
Model	Accuracy
Random Forest	98.59%
XGBoost	99.22%

XGBoost gives the best performance

Covariance and FFT-based features show highest influence

Generated Outputs

Located in the results/ folder:

Confusion_Matrix_RF.png

Confusion_Matrix_XGB.png

RandomForest-Feature-Importance.png

XGBoost-Feature-Importance.png

Technologies Used

Python 3

NumPy

Pandas

Scikit-Learn

XGBoost

Matplotlib

Seaborn

Contributors
Akash A.

Preprocessing & label encoding

Feature selection

RF & XGB model development

Confusion matrices and plots

Methodology, Results, Discussion writing

Sujan A.

Literature review and related work

Introduction writing

Result interpretation

Final report structuring and formatting

License

For academic and research use under the Pattern Recognition course.
