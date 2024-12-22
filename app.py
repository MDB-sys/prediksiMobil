import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Clustering & Linear Regression with Pretrained Models", layout="wide")

# Sidebar untuk memilih mode
st.sidebar.title("Mode Selection")
mode = st.sidebar.selectbox("Choose an Analysis Mode", ["Clustering", "Linear Regression"])

# Fungsi untuk memuat data
@st.cache_data 
def load_data():
    # Contoh dataset
    data = pd.read_csv('toyota.csv')
    m_encoder = LabelEncoder()
    t_encoder = LabelEncoder()
    f_encoder = LabelEncoder()

    data['model'] = m_encoder.fit_transform(data['model'])
    data['transmission'] = t_encoder.fit_transform(data['transmission'])
    data['fuelType'] = f_encoder.fit_transform(data['fuelType'])
    return data

# Fungsi untuk memuat model pickle
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Memuat model Linear Regression dan Clustering
linear_model = load_model('lr.pkl')  # Ganti dengan path model Linear Regression Anda
clustering_model = load_model('kmeans.pkl')  # Ganti dengan path model Clustering Anda

# Load data
data = load_data()

# Tampilkan data di Streamlit
st.write("### Dataset Preview")
st.dataframe(data.head())

# Mode: Clustering
if mode == "Clustering":
    st.write("## Clustering Analysis")
    
    # Input parameter clustering
    n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    features = st.multiselect("Select Features for Clustering", data.columns, default=data.columns)
    
    if len(features) < 2:
        st.warning("Please select at least 2 features for clustering.")
    else:
        # KMeans Clustering (dengan model pickle)
        
        # st.write(data.columns)
        features = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
        data["Cluster"] = clustering_model.predict(data[features])
        
        # Visualisasi clustering
        st.write(f"### Clustering Result with {n_clusters} Clusters")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=data[features[0]], y=data[features[1]], hue=data["Cluster"], palette="viridis", ax=ax
        )
        plt.title("Clustering Visualization")
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        st.pyplot(fig)
        
        # Tampilkan cluster centers
        st.write("### Cluster Centers")
        st.write(pd.DataFrame(clustering_model.cluster_centers_, columns=features))

# Mode: Linear Regression
elif mode == "Linear Regression":
    st.write("## Linear Regression Analysis")

    # Input parameter untuk regresi
    # st.write(data.columns)
    target = st.selectbox("Select Target Variable", data.columns, index=data.columns.get_loc("price"))
    data_without_price = data.drop(columns=['price'])
    predictors = st.multiselect("Select Predictor Variables", data_without_price.columns, default=data_without_price.columns)
    
    
    if len(predictors) == 0:

        st.warning("Please select at least one predictor variable.")
    else:
        # Prediksi menggunakan model Linear Regression dari pickle
        X = data[predictors]
        y = data[target]
        y_pred = linear_model.predict(X)
        
        st.write(X.shape, y.shape)
        # Evaluasi model
        mse = mean_squared_error(y, y_pred)
        st.write(f"### Mean Squared Error: {mse:.2f}")
        
        # Visualisasi hasil prediksi
        st.write("### Prediction vs Actual")
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(y, y_pred, alpha=0.7, color="blue")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", linewidth=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Prediction vs Actual")
        st.pyplot(fig)


