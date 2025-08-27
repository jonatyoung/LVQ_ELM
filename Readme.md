# Comparative Analysis of Artificial Neural Network Architecture for Activity Data Classification in Fitness Tracker Datasets

---

## **Project Overview (Background)**

The development of *wearable devices* such as fitness trackers has enabled continuous collection of physical activity and health-related data. Fitness trackers commonly record information such as step count, heart rate, sleep quality, and types of exercise performed, making them highly valuable in the fields of healthcare, sports, and healthy lifestyles (Suryana, 2021). However, to maximize these benefits, the raw data needs to be processed through accurate classification systems so that patterns of physical activity can be properly recognized.

Artificial Neural Networks (ANN) are one of the most widely used approaches in data classification. Several ANN architectures frequently applied include **Backpropagation**, **Learning Vector Quantization (LVQ)**, and **Extreme Learning Machine (ELM)**. Backpropagation is well known for its strong generalization ability across various datasets (Pratama, 2019; Budianto, 2020). LVQ, on the other hand, is effective in handling data with well-defined patterns since it uses a prototype-based approach (Widianto, 2020). Meanwhile, ELM provides high efficiency in training because input weights are assigned randomly, making the computation time significantly shorter compared to traditional algorithms (Anggoro, 2019; Ashar, 2020).

Nevertheless, the performance of each algorithm is highly influenced by the characteristics of the dataset being used. In the case of the *Fitness Tracker Dataset*, challenges such as data quality, limited sample size, and preprocessing can have a significant impact on classification results (Santoso, 2019). Therefore, it is essential to conduct a comparative analysis of Backpropagation, LVQ, and ELM to determine the most suitable algorithm for recognizing physical activity patterns based on wearable device data.

---

## **Business Understanding**

### **Problem Statement**

1. How does each ANN algorithm (Backpropagation, LVQ, and ELM) perform in classifying activity data in the *Fitness Tracker Dataset*?
2. Which algorithm is the most optimal based on evaluation parameters such as accuracy, precision, recall, and computation time?
3. To what extent are the performance differences among the three algorithms significant in recognizing physical activity patterns?

### **Objective**

* To analyze the performance of Backpropagation, LVQ, and ELM in classifying fitness tracker data.
* To identify the most optimal algorithm for processing wearable device data based on evaluation metrics (accuracy, precision, recall, and computation time).
* To evaluate the significant differences between the three algorithms in order to understand their effectiveness on physical activity data.

### **Proposed Solution**

* Utilize the *Fitness Tracker Dataset* as the experimental dataset.
* Perform data preprocessing (cleaning, normalization, and categorical transformation).
* Train and test the three ANN architectures (Backpropagation, LVQ, and ELM) using evaluation metrics such as accuracy, precision, recall, and computation time.
* Conduct a comparative analysis to determine which algorithm provides the most balanced performance.
* Provide recommendations for further work, such as hyperparameter tuning, dataset expansion, or applying advanced architectures such as CNNs or RNNs to improve classification accuracy.

---

## Dataset Overview

* **Source**: Kaggle – [Fitness Tracker Dataset](https://www.kaggle.com/datasets/smayanj/fitness-tracker-dataset)
* **Number of entries**: 1,000,000 rows
* **Number of features**: 12 columns
* **Author**: smayanj

---

## Dataset Features

### Feature Description

| Column               | Type    | Description                                                                                             |
| -------------------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `user_id`            | int64   | Unique ID for each user                                                                  |
| `date`               | date    | Record date                                                                    |
| `steps`              | int64   | Total number of steps taken in a day                                                                    |
| `calories_burned`    | float64 | Estimated calories burned per day                                                                       |
| `distance_km`        | float64 | Total distance covered (in kilometers)                                                                  |
| `active_minutes`     | int64   | Total minutes of active movement                                                                        |
| `sleep_hours`        | float64 | Total hours of sleep                                                                                    |
| `heart_rate_avg`     | int64   | Average daily heart rate (beats per minute)                                                             |
| `workout_type`       | object  | Type of workout (e.g., running, cycling, yoga, swimming, gym workout) — contains missing values (\~14%) |
| `weather_conditions` | object  | Daily weather condition (e.g., sunny, rainy, fog, snow, clear)                                          |
| `location`           | object  | General user location (city or region)                                                                  |
| `mood`               | object  | User’s self-reported mood (e.g., happy, neutral, stressed, tired)                                       |

---

## 🔍 Data Exploration Results

1. **Duplicate Records**

   * A check was conducted to identify duplicate rows.
   * Any duplicates found were removed to ensure data consistency.

2. **Missing Values**

   * All columns were inspected for missing values.
   * `workout_type` contained missing entries (\~143,120 rows).
   * Rows with missing values were dropped for this study to maintain dataset integrity.

3. **Correlation Analysis (Heatmap)**

   * Strong correlations:

     * `steps` ↔ `distance_km` — as expected, step count strongly determines distance.
     * `active_minutes` ↔ `calories_burned` — longer active time leads to higher energy expenditure.
   * Moderate correlations:

     * `heart_rate_avg` showed a positive relationship with both `active_minutes` and `calories_burned`.
   * Weak / negligible correlations:

     * `sleep_hours` had little to no correlation with physical activity features.
     * Categorical features (`mood`, `weather_conditions`, `workout_type`) did not show linear correlations with numerical variables but may still carry predictive value for classification.

---

##  Summary Dataset

The dataset consists of **1 million records across 10 features**. After removing duplicates and missing values, the dataset is clean and ready for preprocessing. Correlation analysis confirmed strong relationships among core physical activity features (steps, distance, active minutes, calories), while sleep, mood, and weather provide complementary information. This makes the dataset suitable for classification tasks using ANN architectures (Backpropagation, LVQ, and ELM).


---


## Development Team
* I Putu Paramaananda Tanaya
* Muhammad Aldy Naufal Fadhilah 
* Jonathan Young
* Nada Firdaus
