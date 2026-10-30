# ML Assignment 2 – Classification Models with Streamlit

**M.Tech (AIML) – Machine Learning | BITS Pilani WILP**

---

## a. Problem Statement

In the "away from home" (out-of-home) food and drink market, brands and outlets such as coffee shops, takeaways, restaurants and bars often use promotional coupons to attract customers. Sending offers that customers ignore wastes marketing budget, so it is valuable to know in advance who is likely to respond. For foodservice suppliers, this matters too: their business customers (the cafés, takeaways and restaurants they supply) sell more snacks and drinks when their promotions actually bring people through the door.

The goal of this project is to **predict whether a consumer will accept a food or drink coupon** based on their situation (destination, time, weather, who they are travelling with), the type of offer (outlet type, expiry time, distance to the outlet), and their personal profile and eating-out habits.

This is a **binary classification** problem. Five machine learning models are trained and compared on the same data, and the best model is deployed in an interactive Streamlit web application.

---

## b. Dataset Description

- **Dataset name:** In-Vehicle Coupon Recommendation
- **Source:** UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation
- **Original paper:** Wang, T., Rudin, C., Doshi-Velez, F., Liu, Y., Klampfl, E., & MacNeille, P. (2017). *A Bayesian framework for learning rule sets for interpretable classification.* Journal of Machine Learning Research, 18(1), 2357–2393.
- **License:** CC BY 4.0
- **How the data was collected:** Survey on Amazon Mechanical Turk. Respondents were shown a driving scenario and a coupon, and asked whether they would accept it.
- **Number of instances (rows):** 12,684
- **Number of features:** 25 (plus 1 target column)
- **Target column:** `Y` (1 = coupon accepted, 0 = coupon not accepted)
- **Type of problem:** Binary classification
- **Missing values:** Yes – `car` is missing in about 99% of rows; `Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20` and `Restaurant20To50` each have roughly 1–2% missing.

**Feature summary:**

| Feature | Type | Description |
|---|---|---|
| destination | Categorical | Where the person is driving: No Urgent Place, Home, Work |
| passanger | Categorical | Who is in the car: Alone, Friend(s), Kid(s), Partner |
| weather | Categorical | Sunny, Rainy, Snowy |
| temperature | Integer | Outside temperature (30, 55, 80 °F) |
| time | Categorical | Time of day: 7AM, 10AM, 2PM, 6PM, 10PM |
| coupon | Categorical | Outlet type: Coffee House, Carry out & Take away, Bar, Restaurant(<20), Restaurant(20-50) |
| expiration | Categorical | Coupon validity: 2 hours or 1 day |
| gender | Categorical | Female, Male |
| age | Categorical | Age band (below21 to 50plus) |
| maritalStatus | Categorical | Single, Married partner, Unmarried partner, Divorced, Widowed |
| has_children | Binary | 1 = has children, 0 = no children |
| education | Categorical | Highest education level |
| occupation | Categorical | Occupation group |
| income | Categorical | Annual income band |
| car | Categorical | Type of car (mostly missing) |
| Bar | Categorical | How often the person visits a bar per month |
| CoffeeHouse | Categorical | How often the person visits a coffee house per month |
| CarryAway | Categorical | How often the person gets takeaway food per month |
| RestaurantLessThan20 | Categorical | Monthly visits to restaurants with average spend under $20 |
| Restaurant20To50 | Categorical | Monthly visits to restaurants with average spend $20–$50 |
| toCoupon_GEQ5min | Binary | Outlet is 5+ minutes' drive away |
| toCoupon_GEQ15min | Binary | Outlet is 15+ minutes' drive away |
| toCoupon_GEQ25min | Binary | Outlet is 25+ minutes' drive away |
| direction_same | Binary | Outlet is in the same direction as the destination |
| direction_opp | Binary | Outlet is in the opposite direction to the destination |
| **Y (target)** | Binary | 1 = accepted the coupon, 0 = did not accept |

**Class balance:** 7,210 accepted (56.8%) and 5,474 not accepted (43.2%), so the classes are fairly balanced.

**Preprocessing steps:**
- Dropped `car` (about 99% missing), `toCoupon_GEQ5min` (same value in every row) and `direction_opp` (always the opposite of `direction_same`).
- Removed 74 duplicate rows, leaving 12,610 rows and 22 features.
- Split into 80% training (10,088 rows) and 20% test (2,522 rows), stratified on `Y` so both sets keep the same class balance. The test set is saved as `test_data.csv`.
- Built a preprocessing pipeline, fitted on the training data only:
  - Categorical columns (17): missing values filled with the most frequent value, then one-hot encoded.
  - Numeric columns (5): standardised with `StandardScaler`.
- Each model is saved together with this pipeline, so the Streamlit app can use the raw test CSV directly without retraining.

---

## c. GitHub Repository Link

https://github.com/HarithaAIML/ml-classification-streamlit-app

**Live Streamlit App:** https://haritha-coupon-predictor.streamlit.app/

**Repository structure:**

```
ml-classification-streamlit-app/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── test_data.csv          # Test data used for evaluation
├── model_training.ipynb   # Notebook: preprocessing + training
└── model/                 # Saved trained models and preprocessing files
```

---

## d. Models Used

The following five classification models were trained on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6808 | 0.7306 | 0.6984 | 0.7701 | 0.7325 | 0.3417 |
| Decision Tree | 0.6959 | 0.7224 | 0.7078 | 0.7904 | 0.7468 | 0.3726 |
| KNN | 0.6848 | 0.7404 | 0.6978 | 0.7841 | 0.7384 | 0.3490 |
| Naive Bayes | 0.6277 | 0.6678 | 0.6892 | 0.6261 | 0.6562 | 0.2536 |
| Support Vector Machine | **0.7411** | **0.8064** | **0.7462** | **0.8239** | **0.7831** | **0.4674** |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 68.08% accuracy and an AUC of 0.7306, indicating moderate classification performance. Its recall of 77.01% is reasonably good, but its MCC of 0.3417 indicates moderate overall predictive agreement. |
| Decision Tree | Achieved 69.59% accuracy, with a recall of 79.04% and an F1-score of 0.7468. It performs better than Logistic Regression in accuracy, recall, and MCC, but its AUC of 0.7224 is slightly lower. |
| KNN | Achieved 68.48% accuracy and an AUC of 0.7404. Its recall of 78.41% is good, but its accuracy and F1-score are lower than those of the Decision Tree and SVM models. |
| Naive Bayes | Achieved 62.77% accuracy and an AUC of 0.6678, the lowest among all five models. Its recall of 62.61% and F1-score of 0.6562 indicate comparatively weaker classification performance. |
| Support Vector Machine | Achieved the highest accuracy (74.11%), AUC (0.8064), precision (74.62%), recall (82.39%), F1-score (0.7831), and MCC (0.4674). It demonstrates the best overall performance across all six evaluation metrics. |
| **Overall, winner for this dataset?** | Support Vector Machine (SVM) is the best-performing model because it achieves the highest scores across all six metrics, indicating the strongest overall classification performance among the evaluated models. |

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` in the app, select a model, and view the metrics and confusion matrix.
