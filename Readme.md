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


## Development Team
* I Putu Paramaananda Tanaya
* Muhammad Aldy Naufal Fadhilah 
* Jonathan Young
* Nada Firdaus
