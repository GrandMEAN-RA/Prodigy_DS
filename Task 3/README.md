# 📊 Bank Term Deposit Subscription Prediction  
### Machine Learning with Decision Trees (Scikit-Learn Pipeline)

This project builds an interpretable **Decision Tree Classification Model** to predict whether a customer will subscribe to a bank term deposit. It uses a full data preprocessing pipeline with one-hot encoding, pruning, evaluation metrics, and visualizations.

---

## 📌 Project Objectives
- Build an interpretable decision tree model  
- Apply best-practice preprocessing using ColumnTransformer  
- Evaluate the model using multiple metrics  
- Visualize the tree and feature importance  
- Test the effect of removing the *duration* feature  

---

## 📂 Dataset
- **Shape:** 4,119 rows × 21 columns  
- **Features:**  
  - Demographics (age, job, marital status, education)  
  - Call details (contact type, month, day_of_week, duration)  
  - Previous campaign interactions  
  - Economic indicators (euribor3m, emp.var.rate, cons.conf.idx)  
- **Target:** `y` → "yes" or "no" term deposit subscription

---

## 🧠 Model Workflow
1. Split data into train/test  
2. Preprocess:
   - Standardize numeric columns  
   - One-hot encode categorical columns  
3. Train Decision Tree Classifier  
4. Tune depth  
5. Prune tree  
6. Evaluate:
   - Accuracy  
   - Precision/Recall  
   - Confusion Matrix  
   - ROC Curve  
7. Re-train model without duration to compare performance  

---

## 📈 Results Summary
| Model Version | Accuracy | Notes |
|--------------|----------|-------|
| With Duration | ~High | Duration is very predictive |
| Without Duration | Lower | Shows robustness of other features |

---

## 🧩 Visuals Included
- Decision Tree Plot  
- Feature Importance Bar Chart  
- ROC Curve  
- Confusion Matrix  

---

## 🛠 Tech Stack
- Python  
- Pandas  
- Scikit-Learn  
- Matplotlib & Seaborn  
- NumPy  
- Jupyter/Spyder  

---

## ▶️ Run the Project
```bash
pip install -r requirements.txt
python decision_tree_model.py

