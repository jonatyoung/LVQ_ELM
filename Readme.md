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

The dataset consists of **1 million records across 12 features**. After removing duplicates and missing values, the dataset is clean and ready for preprocessing. Correlation analysis confirmed strong relationships among core physical activity features (steps, distance, active minutes, calories), while sleep, mood, and weather provide complementary information. This makes the dataset suitable for classification tasks using ANN architectures (Backpropagation, LVQ, and ELM).

---

## Data Preparation

Before modeling, the dataset underwent the following preprocessing steps:

1. **Data Cleaning**

   * Dropped duplicates and rows with missing values (`workout_type` had \~14% missing values).
   * Ensured each record (`user_id`, `date`) was unique.

2. **Feature Encoding**

   * Categorical variables (`workout_type`, `weather_conditions`, `location`, `mood`) were transformed using **One-Hot Encoding** for input features and **Label Encoding** for the target (`mood`).

3. **Normalization**

   * Numerical features (`steps`, `calories_burned`, `distance_km`, `active_minutes`, `sleep_hours`, `heart_rate_avg`) were standardized using **StandardScaler** to improve convergence in neural network training.

4. **Train-Test Split**

   * Data was split into **80% training** and **20% testing** to ensure robust evaluation.

---

## Modeling

Three Artificial Neural Network (ANN) algorithms were implemented and compared:

### **Backpropagation Neural Network (BPNN)**

* **Architecture**: 2 hidden layers (20 and 10 neurons), sigmoid activation, trained with gradient descent.
* **Output**: Single-node sigmoid for binary classification (extended to multiclass).
* **Training**: 1000 epochs, learning rate = 0.01.

### **Learning Vector Quantization (LVQ)**

* **Mechanism**: Prototype-based classification. Each class represented by prototypes updated iteratively.
* **Setup**: 1 prototype per class, learning rate = 0.1, max iteration = 10.
* **Distance Metric**: Euclidean distance.

### **Extreme Learning Machine (ELM)**

* **Architecture**: Single hidden layer with 100 neurons, sigmoid activation.
* **Training**: Input weights randomized, output weights computed analytically using Moore-Penrose pseudoinverse.
* **Advantage**: Fast training compared to iterative methods.

---

## Evaluation

### Accuracy Comparison

| Model                  | Accuracy (%) |
| ---------------------- | ------------ |
| **ELM**                | 25.05%       |
| **Backpropagation NN** | 25.07%       |
| **LVQ**                | 24.90%       |

→ All models achieved \~25% accuracy, close to **random guessing for 4 classes**.

---

### Confusion Matrices

* **Backpropagation NN**: Predicted all samples as *Neutral*.
* **LVQ**: Predicted all samples as *Tired*.
* **ELM**: Distributed predictions between *Neutral* and *Stressed*, but still failed to classify *Happy* or *Tired* properly.

---

### Classification Report Highlights

* **Backpropagation NN**

  * Recall = 1.00 for *Neutral*, but 0.00 for other moods.
  * Severe class imbalance in predictions.

* **LVQ**

  * Recall = 1.00 for *Tired*, but 0.00 for other moods.
  * Overfitting to a single label.

* **ELM**

  * Slightly better distribution (*Neutral* and *Stressed* classified to some extent).
  * Still very poor precision/recall for *Happy* and *Tired*.

---

## Answer to Problem Statement

1. **How does each ANN algorithm perform?**

   * All three ANN methods (Backpropagation, LVQ, ELM) struggled to classify mood labels correctly, with accuracy stuck around \~25% (random baseline).
   * Backpropagation and LVQ overfit to a single class, while ELM at least attempted to separate two classes (*Neutral* and *Stressed*).

2. **Which algorithm is the most optimal?**

   * In this experiment, **ELM** is the most balanced, with slightly better precision and recall across multiple classes compared to LVQ and Backpropagation, even though overall accuracy is still low.
   * Backpropagation and LVQ completely collapsed into predicting a single mood class.

3. **Are performance differences significant?**

   * Numerically, no. All accuracies are \~25%.
   * Qualitatively, yes: ELM shows marginally better generalization than LVQ and Backpropagation, which were fully biased toward one label.

---

## Key Insights & Future Improvements

* The poor performance suggests:

  * **Target imbalance** (moods may not be equally distributed, even though test set looked balanced).
  * **Model mismatch** (shallow ANNs might not capture the complexity of behavioral patterns).
  * **Feature limitations** (steps, sleep, heart rate, etc., may not strongly determine mood without contextual factors).

* Potential improvements:
  - Try **deeper architectures** (e.g., multi-layer perceptrons with softmax output).
  - Use **regularization** (dropout, weight decay) to prevent collapse into one-class predictions.
  - Apply **class-weighting or oversampling** to handle label imbalance.
  - Consider **ensemble models** or **tree-based methods** (Random Forest, XGBoost) for tabular data.
---

### Conclusion

This study compared the performance of Backpropagation, Learning Vector Quantization (LVQ), and Extreme Learning Machine (ELM) in classifying physical activity data from fitness trackers, with results showing relatively low accuracy of around 25%. The findings suggest that the dataset’s complexity, parameter selection, and possible data noise limited the models’ effectiveness. Despite these limitations, the research highlights the challenges of applying neural networks to wearable device data and provides valuable insights for further exploration. Future work should focus on parameter optimization, improved dataset quality, and advanced models such as CNNs, RNNs, or alternative algorithms like SVM and Random Forest. While performance was below expectations, this study contributes an important foundation for advancing machine learning applications in health and sports technology, particularly in enhancing the analytical capabilities of wearable devices.

---



## References

Pratama, A. (2019). *Jaringan syaraf tiruan algoritma Backpropagation*. Retrieved from [https://media.neliti.com/media/publications/279914-jaringan-syaraf-tiruan-algoritma-backpro-f0165b57.pdf](https://media.neliti.com/media/publications/279914-jaringan-syaraf-tiruan-algoritma-backpro-f0165b57.pdf)

Suryana, F. (2021). *Analisis performa algoritma Backpropagation jaringan syaraf tiruan*. BINUS University. Retrieved from [https://binus.ac.id/bandung/2021/04/analisis-performa-algoritma-backpropagation-jaringan-syaraf-tiruan/](https://binus.ac.id/bandung/2021/04/analisis-performa-algoritma-backpropagation-jaringan-syaraf-tiruan/)

Budianto, I. (2020). Penerapan algoritma Backpropagation pada sistem klasifikasi data. *Jurnal Teknologi Informasi dan Ilmu Komputer, 9*(4), 167–175. Retrieved from [https://jtiik.ub.ac.id/index.php/jtiik/article/view/4806/pdf](https://jtiik.ub.ac.id/index.php/jtiik/article/view/4806/pdf)

Anggoro, E. (2019). *Extreme Learning Machine: Penerapan dan aplikasi*. Retrieved from [https://repository.penerbiteureka.com/media/publications/559194-extreme-learning-machine-penerapan-dan-a-57677ed4.pdf](https://repository.penerbiteureka.com/media/publications/559194-extreme-learning-machine-penerapan-dan-a-57677ed4.pdf)

Ashar, N. M. (2020). *Penerapan Extreme Learning Machine pada klasifikasi data pengguna* (Undergraduate thesis, Universitas Brawijaya). Retrieved from [https://repository.ub.ac.id/id/eprint/13394/1/Nirzha%20Maulidya%20Ashar.pdf](https://repository.ub.ac.id/id/eprint/13394/1/Nirzha%20Maulidya%20Ashar.pdf)

Santoso, D. P. (2019). *Penerapan Extreme Learning Machine pada model prediksi kinerja sistem*. Universitas Airlangga. Retrieved from [https://repository.unair.ac.id/97607/3/3.%20BAB%20I%20PENDAHULUAN.pdf](https://repository.unair.ac.id/97607/3/3.%20BAB%20I%20PENDAHULUAN.pdf)


---


## Development Team
* I Putu Paramaananda Tanaya
* Muhammad Aldy Naufal Fadhilah 
* Jonathan Young
* Nada Firdaus
