# **EEG-Based Emotion Recognition Using Random Forest and XGBoost**

This project presents a complete machine learning pipeline for classifying emotional states from EEG (Electroencephalography) signals using ensemble algorithms. EEG signals contain rich temporal, spectral, and spatial features reflecting underlying cognitive and affective states. By leveraging a curated set of 510 meaningful EEG-derived features and applying robust ensemble learning methods—Random Forest and XGBoost—this system achieves highly accurate emotion classification. The project is designed for academic research, pattern recognition coursework, and EEG-based affective computing applications.

---

## **📌 Overview**

* Dataset: **2132 EEG samples**
* Extracted Features: **2549**, reduced to **510 selected features**
* Emotion Labels: **Negative (0), Neutral (1), Positive (2)**
* Machine Learning Models:

  * **Random Forest – 98.59% accuracy**
  * **XGBoost – 99.22% accuracy**

The system includes preprocessing, label encoding, feature selection, model training, confusion matrix generation, and feature importance visualization. The script is fully automated and accepts any compatible EEG CSV file.

---

## **📁 Project Structure**

```
├── pr.py                      # Main Python script
├── dataset/                   # EEG dataset (samples & features)
├── results/                   # Confusion matrices + importance plots
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## **✨ Features**

* Fully automated machine learning workflow
* Works with any EEG feature CSV (same format)
* Stratified train-test split
* High-performance ensemble models
* Confusion matrix visualizations
* Feature importance ranking
* Reproducible and optimized code

---

## **📊 Dataset Description**

The dataset includes:

* **2132 EEG recordings**
* **2549 engineered features** extracted from:

  * FFT spectral peaks
  * Time-domain statistics
  * Wavelet decompositions
  * Covariance matrices
  * Correlation & signal connectivity metrics

After filtering redundant or low-variance features, **510 high-quality features** were retained for modeling.

---

## **⚙️ Installation**

Clone the repository:

```bash
git clone https://github.com/yourusername/EEG-Emotion-Recognition.git
cd EEG-Emotion-Recognition
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## **▶️ Usage**

Run the main script:

```bash
python pr.py
```

The script will automatically:

1. Load the EEG dataset
2. Encode labels
3. Select 510 features
4. Train Random Forest & XGBoost models
5. Compute accuracy and classification reports
6. Generate:

   * Confusion matrices
   * Feature importance graphs
7. Save all results in the `results/` folder

---

## **🏆 Model Performance**

### **Random Forest**

* Accuracy: **98.59%**

### **XGBoost**

* Accuracy: **99.22%**
* Best overall performer

Key findings:

* Covariance features and FFT spectral bins are the strongest discriminators
* Ensemble models handle high-dimensional EEG signals effectively
* XGBoost provides superior class separation and generalization

---

## **📍 Results (Generated Automatically)**

* `Confusion_Matrix_RF.png`
* `Confusion_Matrix_XGB.png`
* `RandomForest-Feature-Importance.png`
* `XGBoost-Feature-Importance.png`

---

## **🧠 Technologies Used**

* Python 3
* NumPy
* Pandas
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn

---

## **👨‍💻 Contributors**

### **Akash A.**

* Preprocessing, feature selection
* Model implementation (RF & XGB)
* Visualization & evaluation
* Wrote Methodology, Results, Discussion
* Organized Python workflow

### **Sujan A.**

* Literature review
* Introduction & Related Work writing
* Helped interpret model performance
* Final report structuring & editing

---

## **📄 License**

This project is created for academic and research use under the **Pattern Recognition** course at **IIIT Sri City**.

---

