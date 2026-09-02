# Student Academic Failure Risk Prediction

A machine learning project that predicts whether a student is at risk of academic failure, based on behavioral, academic, and lifestyle indicators. The project includes a full data science pipeline (preprocessing, model training, evaluation, and comparison across five classifiers) along with a simple GUI for interactive predictions.

---

##  Project Overview

Academic failure prediction can help institutions identify at-risk students early and intervene with appropriate support. This project:

- Loads and preprocesses a student dataset (encoding categorical features, scaling numeric features, and balancing classes with SMOTE)
- Trains and evaluates **five classification models**:
  - Random Forest
  - Logistic Regression
  - Gaussian Naive Bayes
  - K-Nearest Neighbors (with automatic K selection)
  - Decision Tree
- Compares all models using Accuracy, Precision, Recall, and F1-score
- Visualizes results with confusion matrices and a performance heat map
- Provides a simple GUI for interactive, real-time risk predictions

---

##  Team

This project was built collaboratively by a team of 5:

| Name | Contribution |
|------|--------------|
| Karen | Data preprocessing and Random Forest model and GUI |
| Mayar | Logistic Regression model and deployment |
| Zinab | Naive Bayes model and models comparison |
| Salma | KNN model and presentation |
| Dania | Decision Tree model |

---

## Repository Structure

```
├── StudentsFailureRisk.py     # Main analysis: preprocessing, model training & evaluation
├── gui/                        # GUI application
├── README.md
```

---

##  Dataset

- **Target variable:** `Academic_Failure_Risk` (binary: at risk / not at risk)
- **Features used:** all columns except `Academic_Failure_Risk`, `Student_ID`, `Gender`, `Social_Isolation_Score`, `Physical_Health_Score`, `Physical_Activity_Hours`, `Daily_AI_Tool_Usage_Hours`, `Education_Level`, and `Age` (dropped during preprocessing)
- **Source:** loaded from kaggle 

### Preprocessing Steps

1. Split data into training (80%) and testing (20%) sets, stratified by target
2. Encode categorical columns with `OrdinalEncoder`
3. Scale numeric features with `StandardScaler`
4. Balance the training set with **SMOTE** to address class imbalance

---

##  Models & Evaluation

Each model is trained on the same preprocessed data and evaluated with:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

A final comparison table and heat map summarize all five models side by side, and the best-performing model (by F1-score) is identified automatically.

| Model | Notes |
|-------|-------|
| Random Forest | 150 estimators |
| Logistic Regression | Default parameters |
| Gaussian Naive Bayes | Default parameters |
| K-Nearest Neighbors | Best K chosen by testing K = 1–20 |
| Decision Tree | Max depth = 3 (gini criterion) |

---

##  GUI

A lightweight graphical interface built on top of the trained model lets users enter student feature values and instantly view the predicted academic failure risk.

---

##  Installation & Setup

1. Clone the repository and navigate into the project folder.

2. Install dependencies:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
   ```

3. Run the main analysis:
   ```bash
   python StudentsFailureRisk.py
   ```

4. Launch the GUI from the `gui/` folder to make interactive predictions.

---

##  Results

The script automatically compares all five models on Accuracy, Precision, Recall, and F1-score, visualizes the results in a heat map, and reports the top-performing model by F1-score at the end of the run.
