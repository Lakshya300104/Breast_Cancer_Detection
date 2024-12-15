import pickle
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# CSS for Styling
def add_custom_styles():
    st.markdown(
        """
        <style>
        body {
            background-color: #f8f9fa;
            font-family: Arial, sans-serif;
        }
        h1 {
            text-align: center;
            color: #0073e6;
            font-size: 36px;
            margin-bottom: 20px;
        }
        h2 {
            color: #004085;
            font-size: 24px;
            margin-top: 20px;
        }
        h3 {
            color: #004085;
            font-size: 20px;
        }
        p {
            text-align: justify;
            font-size: 16px;
            line-height: 1.6;
        }
        .stButton>button {
            background-color: #0073e6;
            color: white;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Load models
def load_logistic_regression():
    data = pd.read_csv('data.csv')
    data = data.drop(columns='id', axis=1)
    x = data.drop(columns='label', axis=1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model

def load_naive_bias():
    data = pd.read_csv('data.csv')
    data = data.drop(columns='id', axis=1)
    x = data.drop(columns='label', axis=1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    model = MultinomialNB()
    model.fit(x_train, y_train)
    return model


# Main App
def main_app():
    add_custom_styles()
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Introduction", "Breast Cancer Analysis"])

    if page == "Introduction":
        introduction()
    elif page == "Breast Cancer Analysis":
        breast_cancer_analysis()

# Introduction Page
# Introduction Page
def introduction():
    st.markdown("<h1>Welcome to the Breast Cancer Prediction App</h1>", unsafe_allow_html=True)

    # Add styled text content
    st.write(
        """
        Breast cancer is one of the most prevalent cancers globally and affects millions of lives each year. It occurs 
        due to the abnormal growth of cells in the breast tissues, leading to the formation of **tumors**. Detecting 
        and diagnosing these tumors early can drastically improve the chances of successful treatment and survival.

        There are **two primary types of tumors** based on their characteristics:
        """
    )

    # Benign Tumors Section
    st.markdown("### Benign Tumor")
    st.write(
        """
        A **benign tumor** is **non-cancerous** and typically less harmful than malignant tumors. Key characteristics of benign tumors include:
        - **Non-cancerous**: The tumor does not invade nearby tissues or spread to other parts of the body.
        - **Capsulated**: It remains confined within a well-defined boundary.
        - **Non-invasive**: It does not damage or infiltrate surrounding tissues.
        - **Slow growing**: These tumors grow at a slow and controlled rate.
        - **No metastasis**: Benign tumors do not spread to distant organs.
        - **Cells are normal**: The cells appear normal under a microscope, with a regular structure and nucleus.

        While benign tumors are generally harmless, regular monitoring is recommended to ensure they do not change over time.
        """
    )

    # Malignant Tumors Section
    st.markdown("### Malignant Tumor")
    st.write(
        """
        A **malignant tumor** is **cancerous** and poses significant health risks. Malignant tumors have the following characteristics:
        - **Cancerous**: The tumor grows uncontrollably and invades nearby tissues.
        - **Non-capsulated**: The tumor lacks a defined boundary, making it more aggressive.
        - **Fast growing**: Malignant tumors proliferate quickly compared to benign tumors.
        - **Metastasis**: These tumors can spread (metastasize) to other parts of the body via the bloodstream or lymphatic system.
        - **Abnormal cells**: Cells have large, dark nuclei and irregular shapes when observed under a microscope.

        Malignant tumors require immediate medical attention as they can damage vital organs and spread rapidly throughout the body.
        """
    )

    # Overview of App
    st.write(
        """
        ### About This App:
        This application utilizes **machine learning models** to classify breast tumors as either **benign** or 
        **malignant** based on input clinical features such as radius, texture, perimeter, and smoothness of the tumor. 

        ### Models Used:
        1. **Naive Bayes**:
           - A probabilistic classifier that works efficiently for high-dimensional data.
           - Assumes independence among features, making it simple yet effective for classification tasks.

        2. **Logistic Regression**:
           - A statistical model used for binary classification problems.
           - It predicts the probability of a tumor being malignant or benign with high accuracy.

        **How It Works**:
        - Input the required clinical features extracted from medical imaging.
        - Select a machine learning model (Logistic Regression or Naive Bayes).
        - Receive a prediction indicating whether the tumor is **malignant** or **benign**.

        ### Disclaimer:
        This application is designed for **educational purposes only** and should not be used as a substitute for 
        professional medical advice, diagnosis, or treatment.
        """
    )

    # Add color styling to the content
    st.markdown(
        """
        <style>
        h1 {
            text-align: center;
            color: #0073e6;
            font-size: 36px;
            margin-bottom: 20px;
        }
        h2, h3 {
            color: #004085;
        }
        p {
            text-align: justify;
            font-size: 16px;
            line-height: 1.6;
        }
        .stMarkdown {
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Breast Cancer Analysis Page
def breast_cancer_analysis():
    st.markdown("<h1>Breast Cancer Prediction System</h1>", unsafe_allow_html=True)

    # Model selection
    st.write("### Choose a Machine Learning Model:")
    models_av = ["Select One", "Logistic Regression", "Naive Bayes"]
    choice = st.selectbox("Which Model do you want to use?", models_av)

    # When "Select One" is chosen
    if choice == "Select One":
        st.write(
            """
            ### Model Features:
            #### 1. Logistic Regression:
            - **Use Case**: Logistic regression is effective for binary classification tasks like breast cancer 
              detection.
            - **Advantages**:
              - Offers probabilistic predictions, helping users understand the likelihood of malignancy.
              - Well-suited for structured medical data.

            #### 2. Naive Bayes:
            - **Use Case**: Particularly useful for high-dimensional datasets and simple binary classification.
            - **Advantages**:
              - Computationally efficient.
              - Assumes feature independence, simplifying computations for large datasets.

            Both models are extensively used in healthcare for predictive analytics and diagnostic support.
            """
        )

    # When a specific model is selected
    elif choice in ["Logistic Regression", "Naive Bayes"]:
        st.write(
            """
            ### Input Tumor Features:
            Each feature corresponds to a clinical or radiological characteristic of the tumor. 
            These features are crucial for machine learning models to classify tumors accurately.
            """
        )
        model = load_logistic_regression() if choice == "Logistic Regression" else load_naive_bias()

        # Input fields for tumor features
        st.write("#### Enter the Required Medical Features:")
        radius_mean = st.number_input("Radius Mean:", min_value=0.0, step=0.1, format="%.2f")
        texture_mean = st.number_input("Texture Mean:", min_value=0.0, step=0.1, format="%.2f")
        perimeter_mean = st.number_input("Perimeter Mean:", min_value=0.0, step=0.1, format="%.2f")
        area_mean = st.number_input("Area Mean:", min_value=0.0, step=0.1, format="%.2f")
        smoothness_mean = st.number_input("Smoothness Mean:", min_value=0.0, step=0.001, format="%.3f")
        compactness_mean = st.number_input("Compactness Mean:", min_value=0.0, step=0.001, format="%.3f")
        concavity_mean = st.number_input("Concavity Mean:", min_value=0.0, step=0.001, format="%.3f")
        concave_points_mean = st.number_input("Concave Points Mean:", min_value=0.0, step=0.001, format="%.3f")
        symmetry_mean = st.number_input("Symmetry Mean:", min_value=0.0, step=0.001, format="%.3f")
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean:", min_value=0.0, step=0.001, format="%.3f")
        radius_se = st.number_input("Radius SE:", min_value=0.0, step=0.1, format="%.3f")
        texture_se = st.number_input("Texture SE:", min_value=0.0, step=0.1, format="%.3f")
        perimeter_se = st.number_input("Perimeter SE:", min_value=0.0, step=0.1, format="%.3f")
        area_se = st.number_input("Area SE:", min_value=0.0, step=0.1, format="%.3f")
        smoothness_se = st.number_input("Smoothness SE:", min_value=0.0, step=0.001, format="%.3f")
        compactness_se = st.number_input("Compactness SE:", min_value=0.0, step=0.001, format="%.3f")
        concavity_se = st.number_input("Concavity SE:", min_value=0.0, step=0.001, format="%.3f")
        concave_points_se = st.number_input("Concave Points SE:", min_value=0.0, step=0.001, format="%.3f")
        symmetry_se = st.number_input("Symmetry SE:", min_value=0.0, step=0.001, format="%.3f")
        fractal_dimension_se = st.number_input("Fractal Dimension SE:", min_value=0.0, step=0.001, format="%.3f")
        radius_worst = st.number_input("Radius Worst:", min_value=0.0, step=0.1, format="%.3f")
        texture_worst = st.number_input("Texture Worst:", min_value=0.0, step=0.1, format="%.3f")
        perimeter_worst = st.number_input("Perimeter Worst:", min_value=0.0, step=0.1, format="%.3f")
        area_worst = st.number_input("Area Worst:", min_value=0.0, step=0.1, format="%.3f")
        smoothness_worst = st.number_input("Smoothness Worst:", min_value=0.0, step=0.001, format="%.3f")
        compactness_worst = st.number_input("Compactness Worst:", min_value=0.0, step=0.001, format="%.3f")
        concavity_worst = st.number_input("Concavity Worst:", min_value=0.0, step=0.001, format="%.3f")
        concave_points_worst = st.number_input("Concave Points Worst:", min_value=0.0, step=0.001, format="%.3f")
        symmetry_worst = st.number_input("Symmetry Worst:", min_value=0.0, step=0.001, format="%.3f")
        fractal_dimension_worst = st.number_input("Fractal Dimension Worst:", min_value=0.0, step=0.001, format="%.3f")

        # Prediction
        input_data = pd.DataFrame([{
            'radius_mean': radius_mean,
            'texture_mean': texture_mean,
            'perimeter_mean': perimeter_mean,
            'area_mean': area_mean,
            'smoothness_mean': smoothness_mean,
            'compactness_mean': compactness_mean,
            'concavity_mean': concavity_mean,
            'concave points_mean': concave_points_mean,
            'symmetry_mean': symmetry_mean,
            'fractal_dimension_mean': fractal_dimension_mean,
            'radius_se': radius_se,
            'texture_se': texture_se,
            'perimeter_se': perimeter_se,
            'area_se': area_se,
            'smoothness_se': smoothness_se,
            'compactness_se': compactness_se,
            'concavity_se': concavity_se,
            'concave points_se': concave_points_se,
            'symmetry_se': symmetry_se,
            'fractal_dimension_se': fractal_dimension_se,
            'radius_worst': radius_worst,
            'texture_worst': texture_worst,
            'perimeter_worst': perimeter_worst,
            'area_worst': area_worst,
            'smoothness_worst': smoothness_worst,
            'compactness_worst': compactness_worst,
            'concavity_worst': concavity_worst,
            'concave points_worst': concave_points_worst,
            'symmetry_worst': symmetry_worst,
            'fractal_dimension_worst': fractal_dimension_worst,
        }])

        if st.button("Predict"):
            if radius_mean == 0.0 or texture_mean == 0.0 or perimeter_mean == 0.0:
                st.warning("Please provide all the required feature values.")
            else:
                with st.spinner("Analyzing..."):
                    prediction = model.predict(input_data)
                if prediction[0] == 'M':
                    st.success(
                        """
                        The tumor is classified as **Malignant (cancerous)**.
                        Immediate medical attention is recommended. Please consult a healthcare provider for further 
                        evaluation and treatment options.
                        """
                    )
                else:
                    st.success(
                        """
                        The tumor is classified as **Benign (non-cancerous)**.
                        Follow-up is recommended to monitor changes over time.
                        """
                    )

# Run the app
if __name__ == "__main__":
    main_app()
