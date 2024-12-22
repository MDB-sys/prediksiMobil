import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Set the title of the app
st.title('Clustering and Linear Aggregation Dashboard')

# Upload dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Load and display data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")

    df.drop_duplicates()

    

    # ubah euro ke rp
    nilai_tukar_euro_ke_rupiah = 16974
    df['price'] = df['price'] * nilai_tukar_euro_ke_rupiah
    df['tax'] = df['tax'] * nilai_tukar_euro_ke_rupiah




    st.dataframe(df.head())

    # Handle missing values if any
    if df.isnull().sum().sum() > 0:
        st.warning("Dataset contains missing values. We will fill them with mean values.")
        df.fillna(df.mean(), inplace=True)

    # Show the list of columns
    st.sidebar.subheader("Columns for Clustering")
    features = st.sidebar.multiselect("Select Features", df.columns.tolist())
    
    if len(features) > 1:
        st.write(f"Selected features: {features}")

        # Step 1: Data Preprocessing
        # Standardizing the data
        m_encoder = LabelEncoder()
        t_encoder = LabelEncoder()
        f_encoder = LabelEncoder()

        df['model'] = m_encoder.fit_transform(df['model'])
        df['transmission'] = t_encoder.fit_transform(df['transmission'])
        df['fuelType'] = f_encoder.fit_transform(df['fuelType'])

        featurs = ["model", "year", "transmission", "mileage", "fuelType", "tax", "mpg", "engineSize"]
        X= df[featurs]
        y = df['price']

        scaler = StandardScaler()
        
        df_scaled = scaler.fit_transform(df[features])

        # Step 2: K-Means Clustering
        st.subheader("K-Means Clustering")

        # Number of clusters
        k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)

        # Apply KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df_scaled)

        # Display clustering result
        st.write(f"K-Means Clustering Results (k={k}):")
        st.dataframe(df[['Cluster'] + features].head())

        # Silhouette Score
        silhouette_avg = silhouette_score(df_scaled, df['Cluster'])
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")

        # Visualize Clusters (PCA for dimensionality reduction)
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(df_scaled)
        df_pca = pd.DataFrame(pca_components, columns=['PCA 1', 'PCA 2'])

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PCA 1', y='PCA 2', hue=df['Cluster'], palette='viridis', data=df_pca, s=100, alpha=0.7)
        plt.title("K-Means Clustering Visualization (PCA Reduced)")
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        st.pyplot(plt)

        # Step 2: Linear Aggregation



        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.subheader("Linear Aggregation")

        

        aggregation_feature = st.sidebar.selectbox("Select Feature for Aggregation", df.columns.tolist())
        aggregation_method = st.sidebar.selectbox("Select Aggregation Method", ['Mean', 'Sum'])

        if aggregation_method == 'Mean':
            aggregated_data = df.groupby('Cluster')[aggregation_feature].mean()
        else:
            aggregated_data = df.groupby('Cluster')[aggregation_feature].sum()

        st.write(f"Aggregated {aggregation_method} by Cluster:")
        st.dataframe(aggregated_data)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred = lr.predict(X_test)

        plt.scatter(y_test, pred, color='blue', label='Prediksi')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Garis Ideal')
        plt.xlabel('Nilai Asli (Sales)')
        plt.ylabel('Nilai Prediksi (Sales)')
        plt.title('Prediksi vs Nilai Asli (Linear Regression)')
        plt.legend()
        plt.show()
        st.pyplot(plt)

else:
    st.warning("Please upload a CSV file to proceed.")
