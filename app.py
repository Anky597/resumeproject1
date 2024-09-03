import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA

# Load the pre-trained Mean Shift model
with open('mean_shift_model.pkl', 'rb') as file:
    mean_shift_model = pickle.load(file)

# Function to process user input
def process_input(data):
    # Ensure that the input data is in the same format as used for training
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Streamlit application
st.title('User Segmentation Predictor')

st.sidebar.header('Upload your CSV file')
uploaded_file = st.sidebar.file_uploader(
    "Drag and drop file here",
    type=['csv']
)
st.write(
    """ Creatd by Aniket Anil Nikam """
)

st.markdown("""
**Expected CSV File Format:**

Your CSV file should contain the following columns:
- `User_ID`
- `Age`
- `Daily_Usage_Time (minutes)`
- `Posts_Per_Day`
- `Likes_Received_Per_Day`
- `Comments_Received_Per_Day`
- `Messages_Sent_Per_Day`
- `Gender_Female`
- `Gender_Male`
- `Gender_Marie`
- `Gender_Non-binary`
- `Platform_Instagram`
- `Platform_LinkedIn`
- `Platform_Snapchat`
- `Platform_Telegram`
- `Platform_Twitter`
- `Platform_Whatsapp`
- `Dominant_Emotion_Anxiety`
- `Dominant_Emotion_Boredom`
- `Dominant_Emotion_Happiness`
- `Dominant_Emotion_Neutral`
- `Dominant_Emotion_Sadness`
""")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    # Display uploaded data
    st.write("Uploaded Data Preview")
    st.write(data.head())
    
    # Process the data
    features = data.drop(columns=['User_ID'])
    processed_data = process_input(features)
    
    # Predict clusters
    clusters = mean_shift_model.fit_predict(processed_data)
    
    # Add cluster predictions to the dataframe
    data['Cluster'] = clusters
    
    # Display results
    st.write("Prediction Results")
    st.write(data.head())
    
    # Evaluate the clustering
    silhouette_avg = silhouette_score(processed_data, clusters)
    st.write("Silhouette Score: ", silhouette_avg)
    
    # Cluster Visualization
    st.write("Cluster Visualization:")
    
    # 2D Scatter Plot if data has exactly two features for simplicity
    if processed_data.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(processed_data[:, 0], processed_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('2D Scatter Plot of Clusters')
        st.pyplot(fig)
    else:
        # Implement 3D scatter plot if data is 3D or use PCA for dimensionality reduction
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(processed_data)
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=clusters, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D Scatter Plot of Clusters')
        st.pyplot(fig)

    # Cluster Profiles
    st.write("Cluster Profiles:")
    
    # Display average values for each cluster
    cluster_profiles = data.groupby('Cluster').mean()
    st.write(cluster_profiles)
    # Dominant Emotion Happiness by Cluster
    if 'Dominant_Emotion_Happiness' in data.columns:
        if data['Dominant_Emotion_Happiness'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(data=data, x='Cluster', y='Dominant_Emotion_Happiness', palette='viridis', ax=ax)
            ax.set_title('Average Dominant Emotion Happiness by Cluster')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Dominant Emotion Happiness')
            st.pyplot(fig)
        else:
            st.write("No data available for Dominant_Emotion_Happiness.")
    
    # Daily Usage Time by Cluster
    if 'Daily_Usage_Time (minutes)' in data.columns:
        if data['Daily_Usage_Time (minutes)'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=data, x='Cluster', y='Daily_Usage_Time (minutes)', palette='coolwarm', ax=ax)
            ax.set_title('Daily Usage Time by Cluster')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Daily Usage Time (minutes)')
            st.pyplot(fig)
        else:
            st.write("No data available for Daily_Usage_Time (minutes).")
    
    # Number of Users in Each Cluster
    st.write("Number of Users in Each Cluster:")
    cluster_counts = data['Cluster'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='plasma', ax=ax)
    ax.set_title('Number of Users in Each Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Users')
    st.pyplot(fig)
    
    # Platform Usage Insights
    st.write("Platform Usage Insights:")
    platform_columns = [col for col in data.columns if 'Platform_' in col]
    
    if platform_columns:
        for col in platform_columns:
            if data[col].notna().any():  # Ensure column has non-NaN values
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.boxplot(data=data, x='Cluster', y=col, palette='coolwarm', ax=ax)
                ax.set_title(f'{col} by Cluster')
                ax.set_xlabel('Cluster')
                ax.set_ylabel(col)
                st.pyplot(fig)
            else:
                st.write(f"No data available for {col}")
    else:
        st.write("No platform usage columns found in the dataset.")
    
    # Behavioral Analysis
    st.write("Behavioral Analysis:")
    behavior_columns = ['Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day']
    for col in behavior_columns:
        if col in data.columns:
            if data[col].notna().any():  # Ensure column has non-NaN values
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.boxplot(data=data, x='Cluster', y=col, palette='magma', ax=ax)
                ax.set_title(f'{col} by Cluster')
                ax.set_xlabel('Cluster')
                ax.set_ylabel(col)
                st.pyplot(fig)
            else:
                st.write(f"No data available for {col}")
    
    # Emotional Analysis
    st.write("Emotional Analysis:")
    emotion_columns = [col for col in data.columns if 'Dominant_Emotion_' in col]
    if emotion_columns:
        for col in emotion_columns:
            if data[col].notna().any():  # Ensure column has non-NaN values
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(data=data, x='Cluster', y=col, palette='magma', ax=ax)
                ax.set_title(f'Average {col} by Cluster')
                ax.set_xlabel('Cluster')
                ax.set_ylabel(col)
                st.pyplot(fig)
            else:
                st.write(f"No data available for {col}")

    # Additional Insights
    st.write("Additional Insights:")
    
    
