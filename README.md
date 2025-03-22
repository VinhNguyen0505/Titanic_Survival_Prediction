
# 🚢 Titanic Survival Prediction

This project builds multiple machine learning models—including **Logistic Regression**, **Random Forest**, **XGBoost**, and a **Neural Network (TensorFlow/Keras)**—to predict passenger survival from the Titanic dataset. It includes **data preprocessing**, **model training**, and **evaluation** using multiple metrics.

---

## 📊 Dataset

The dataset used is the [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data), and is loaded from a public source:
```
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

---

## 🔧 Features Used
- `Pclass`
- `Sex` (encoded)
- `Age`
- `Fare`
- `Embarked` (encoded)
- `SibSp`
- `Parch`

---

## 🧠 Models Trained
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Neural Network** using TensorFlow/Keras

---

## 🧼 Preprocessing Steps
- Filled missing `Age` with the median.
- Dropped `Cabin` due to excessive missing data.
- Encoded categorical features (`Sex`, `Embarked`) numerically.
- Normalized features for the Neural Network using `StandardScaler`.

---

## 📈 Evaluation Metrics
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-score)**

---

## 🔍 Neural Network Architecture
- Input Layer: 7 Features
- Hidden Layers: 
  - Dense(16), ReLU
  - Dense(8), ReLU
- Output Layer: 
  - Dense(1), Sigmoid (for binary classification)

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/VinhNguyen0505/Titanic_Survival_Prediction.git
cd Titanic_Survival_Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Python script:
```bash
python titanic_model.py
```

---

## 📦 Requirements

Include this in your `requirements.txt`:
```
pandas
scikit-learn
xgboost
tensorflow
matplotlib
```

---

## 📊 Sample Output
```
Neural Network Accuracy: 0.84
Random Forest Accuracy: 0.87
XGBoost Accuracy: 0.89
```

> Bonus: Add accuracy plots or model comparison if available!

---

## ✨ Author

**Vinh Nguyen**  
- Portfolio: [vinhnguyen0505.github.io](https://vinhnguyen0505.github.io)  
- GitHub: [@VinhNguyen0505](https://github.com/VinhNguyen0505)  
- LinkedIn: [linkedin.com/in/vinh-nguyen](https://linkedin.com/in/vinh-nguyen)

---

## 🔗 License
This project is open-source and free to use for learning and educational purposes.
