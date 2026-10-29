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

**Preprocessing steps:**
- (to be added after preprocessing – missing values, encoding, scaling, train/test split)

---

## c. GitHub Repository Link

(to be added)

**Live Streamlit App:** (to be added)

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
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | (to be added) |
| Decision Tree | (to be added) |
| KNN | (to be added) |
| Naive Bayes | (to be added) |
| Support Vector Machine | (to be added) |
| **Overall, winner for this dataset?** | (to be added) |

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` in the app, select a model, and view the metrics and confusion matrix.
