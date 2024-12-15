# **Breast Cancer Detection System** 🎯  
*Leveraging Machine Learning for Accurate Tumor Classification*

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Technologies Used](#technologies-used)
4. [Machine Learning Models](#machine-learning-models)
5. [Dataset](#dataset)
6. [Setup Guide](#setup-guide)
7. [Usage](#usage)
8. [Demo](#demo)
9. [Learnings and Challenges](#learnings-and-challenges)
10. [Acknowledgments](#acknowledgments)

---

## 🚀 **Introduction**  
Breast cancer is one of the most prevalent cancers globally, and early detection is critical to improving survival rates. This project focuses on building a **Machine Learning-based Prediction System** that classifies breast tumors as either:  
- **Benign (non-cancerous)**  
- **Malignant (cancerous)**  

Using clinical features extracted from tumor images, the system delivers **accurate predictions** through two pre-trained machine learning models: **Logistic Regression** and **Naive Bayes**.

The app is built using **Streamlit**, providing a clean, intuitive interface for real-time predictions.

---

## 🔑 **Key Features**  
- **Real-time Predictions**: Input tumor features and instantly receive the tumor classification.  
- **Multiple Models**: Compare results between Logistic Regression and Naive Bayes models.  
- **Pre-trained Models**: Models are trained on the **Breast Cancer Wisconsin Dataset** for efficiency.  
- **User-Friendly Interface**: Streamlit-powered web app with a clean design for easy interaction.  
- **Seamless Deployment**: Accessible anywhere via **Streamlit Cloud**.

---

## 🛠️ **Technologies Used**  
- **Python**: Main programming language.  
- **Scikit-learn**: For model development and training.  
- **Pandas**: For data preprocessing and manipulation.  
- **Streamlit**: For building the interactive web app.  
- **Pickle**: To save and load pre-trained models.  

---

## 🔍 **Machine Learning Models**  

### 1. **Logistic Regression**  
- A statistical model used for binary classification.  
- Provides probabilities of a tumor being **Benign** or **Malignant**.  
- Highly interpretable and effective for structured medical datasets.  

### 2. **Naive Bayes**  
- A probabilistic classifier based on **Bayes' Theorem**.  
- Assumes independence between features, simplifying computations.  
- Fast and efficient for high-dimensional data.  

Both models are trained, evaluated, and saved for real-time predictions.

---

## 📊 **Dataset**  
- The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**.  
- It contains **30 numerical features** extracted from tumor images (e.g., `radius_mean`, `texture_mean`, `area_worst`).  
- Target Label:  
   - `B` → **Benign**  
   - `M` → **Malignant**  

---

## 💻 **Setup Guide**

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
```

### **2. Install Dependencies**  
Make sure you have Python installed (Python 3.8 or above). Install the required libraries:  
```bash
pip install -r requirements.txt
```

### **3. Run the Application**  
Run the Streamlit app locally:  
```bash
streamlit run App_Deploy.py
```

### **4. Access the App**  
The app will be available at:  
```plaintext
http://localhost:8501
```

---

## 🚀 **Usage**  

1. **Select the Model**: Choose between Logistic Regression and Naive Bayes.  
2. **Input Features**: Provide tumor features such as `radius_mean`, `texture_mean`, `concavity`, etc.  
3. **Predict**: Click on the **Predict** button to get the classification result.  
4. **Interpret the Result**:  
   - **Benign** → Non-cancerous tumor.  
   - **Malignant** → Cancerous tumor requiring medical attention.  

---

## 🎥 **Demo**  
- **Live Demo**: [Working Streamlit App Link](https://breastcancerdetection-5lwexcetuhcrfng6n8rjwh.streamlit.app/)  
---

## 📈 **Learnings and Challenges**  
### **What I Learned**:  
- Training and saving machine learning models for real-time usage.  
- Building interactive web applications using **Streamlit**.  
- Understanding the medical significance of tumor features and their impact on classification.  
- Deploying projects on **Streamlit Cloud** for global accessibility.

### **Challenges Faced**:  
- Ensuring feature compatibility between training and prediction.  
- Optimizing model performance without overfitting.  
- Designing a user-friendly interface for a non-technical audience.  

---

## 🙌 **Acknowledgments**  
- **Scikit-learn** for simplifying machine learning model development.  
- **Streamlit** for enabling fast and intuitive app creation.  
- The creators of the **Breast Cancer Wisconsin Dataset** for providing invaluable data.

---

## 🧩 **Future Scope**  
- Integrating more advanced models like **Random Forest** and **XGBoost** for improved accuracy.  
- Adding visualizations to better interpret model predictions.  
- Expanding the app to include additional cancer diagnostic datasets.

---

🔗 **Connect with Me**:  
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/lakshya-arora-76a567259/)  
- GitHub: [Your GitHub Profile](https://www.linkedin.com/in/lakshya-arora-76a567259/)
  
---

