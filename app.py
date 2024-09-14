import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import skfuzzy as fuzz
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


# Function to load data
@st.cache_data
def load_data():
    return pd.read_csv('features.csv')

# Load data
df = load_data()

# App title
st.title('Chess Clustering Analysis')

st.write("""
    This goal of this project is to cluster the players in each match to one of the 2 clusters, which are bad players and good players.\n
    The higher the ratings of the players, the better the players are.
""")

# Display the image
st.image('chess.jpg', use_column_width=True)

# Display data
st.write("") 
st.subheader("Data")
st.dataframe(df.head(), use_container_width=True)  # Make the DataFrame use the full container width
features = df[['white_rating', 'black_rating']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Sidebar for model selection
model_type = st.sidebar.selectbox("Select Model", ["Fuzzy C-Means Clustering", "Agglomerative Clustering", "BIRCH Clustering", "Gaussian Mixture Model (GMM)", "K Means Clustering"])

if model_type == "Fuzzy C-Means Clustering":
    # Sidebar for parameters
    num_clusters = 2  # Set to 2 clusters
    fuzz_m = st.sidebar.slider("Fuzziness Parameter (m)", min_value=1.1, max_value=3.0, value=2.0, step=0.1)
    max_iter = st.sidebar.slider("Max Iterations", min_value=100, max_value=2000, value=1000)
    error = st.sidebar.slider("Error Tolerance", min_value=0.001, max_value=0.01, value=0.005)


    # Prepare features
    X = X_scaled.T




    # Input and Prediction Section
    st.write("")
    st.subheader("Predict Cluster for New Data")

    white_rating = st.number_input("Enter White Player Rating", min_value=float(df['white_rating'].min()), max_value=float(df['white_rating'].max()))
    black_rating = st.number_input("Enter Black Player Rating", min_value=float(df['black_rating'].min()), max_value=float(df['black_rating'].max()))
    
    # Metrics and Evaluation
    if st.button("Predict Cluster"):
        # Perform Fuzzy C-Means clustering
        cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
            X,
            num_clusters,
            m=fuzz_m,
            error=error,
            maxiter=max_iter
        )


        # Analyze results
        membership = np.argmax(u, axis=0)
        df['Cluster'] = membership

        
        user_data = np.array([[white_rating, black_rating]])
        user_data_scaled = scaler.transform(user_data).T  # Transpose if necessary
        user_membership = fuzz.cluster.cmeans_predict(user_data_scaled, cntr, m=fuzz_m, error=error, maxiter=max_iter)[0]
        predicted_cluster = np.argmax(user_membership)
        st.write(f'The new data point belongs to Cluster {predicted_cluster}')

            # Plot the clusters
        fig, ax = plt.subplots()
        sns.scatterplot(x='white_rating', y='black_rating', hue='Cluster', palette='viridis', data=df, ax=ax)
        ax.set_xlabel('White Player Rating')
        ax.set_ylabel('Black Player Rating')
        ax.set_title('Player Clustering based on Ratings')
        st.pyplot(fig)

        # Metric scores
        silhouette_avg = silhouette_score(X_scaled, membership)
        dbi = davies_bouldin_score(X_scaled, membership)
        calinski_harabasz = calinski_harabasz_score(X_scaled, membership)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Silhouette Score", value=f"{silhouette_avg:.2f}")

        with col2:
            st.metric(label="Davies-Bouldin Index", value=f"{dbi:.2f}")

        with col3:
            st.metric(label="Calinski-Harabasz Index", value=f"{calinski_harabasz:.2f}")

        # Analyze the size of each cluster
        cluster_sizes = df['Cluster'].value_counts()
        st.write("Cluster Sizes")
        st.write(cluster_sizes)


        # Range of potential clusters for evaluation
        range_n_clusters = list(range(2, 10))

        # Metric scores for each cluster count
        silhouette_scores = []
        dbi_scores = []
        calinski_harabasz_scores = []

        # Evaluate different numbers of clusters
        for n_clusters in range_n_clusters:
            cntr_temp, u_temp, _, _, _, _, _ = fuzz.cmeans(X, n_clusters, m=fuzz_m, error=error, maxiter=max_iter)
            membership_temp = np.argmax(u_temp, axis=0)

            # Compute metrics
            silhouette_scores.append(silhouette_score(X_scaled, membership_temp))
            dbi_scores.append(davies_bouldin_score(X_scaled, membership_temp))
            calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, membership_temp))

        # Plot Silhouette Scores
        fig_silhouette, ax_silhouette = plt.subplots(figsize=(4, 3))
        ax_silhouette.plot(range_n_clusters, silhouette_scores, marker='o')
        ax_silhouette.set_title('Silhouette Score for Various Numbers of Clusters')
        ax_silhouette.set_xlabel('Number of Clusters')
        ax_silhouette.set_ylabel('Silhouette Score')
        st.pyplot(fig_silhouette)

        # Plot Davies-Bouldin Index
        fig_dbi, ax_dbi = plt.subplots(figsize=(4, 3))
        ax_dbi.plot(range_n_clusters, dbi_scores, marker='o')
        ax_dbi.set_title('Davies-Bouldin Index for Various Numbers of Clusters')
        ax_dbi.set_xlabel('Number of Clusters')
        ax_dbi.set_ylabel('Davies-Bouldin Index')
        st.pyplot(fig_dbi)

        # Plot Calinski-Harabasz Index
        fig_calinski, ax_calinski = plt.subplots(figsize=(4, 3))
        ax_calinski.plot(range_n_clusters, calinski_harabasz_scores, marker='o')
        ax_calinski.set_title('Calinski-Harabasz Index for Various Numbers of Clusters')
        ax_calinski.set_xlabel('Number of Clusters')
        ax_calinski.set_ylabel('Calinski-Harabasz Index')
        st.pyplot(fig_calinski)

        # Boxplot for white_rating across clusters
        st.write("") 
        st.subheader("Cluster Summary Statistics")

        # Boxplot for white_rating across clusters
        fig_white, ax_white = plt.subplots()
        sns.boxplot(x='Cluster', y='white_rating', data=df, ax=ax_white)
        ax_white.set_title('White Rating Distribution Across Clusters')
        st.pyplot(fig_white)

        # Boxplot for black_rating across clusters
        fig_black, ax_black = plt.subplots()
        sns.boxplot(x='Cluster', y='black_rating', data=df, ax=ax_black)
        ax_black.set_title('Black Rating Distribution Across Clusters')
        st.pyplot(fig_black)

elif model_type == "Agglomerative Clustering":
    # Sidebar for parameters
    n_clusters = 2
    linkage_type = st.sidebar.selectbox("Select Linkage Type", ["ward", "complete", "average", "single"])

    # Perform Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage=linkage_type)
    labels = agg_clustering.fit_predict(X_scaled)

    # Add the cluster labels to your DataFrame
    df['Cluster'] = labels

    # Visualize the clusters using a scatter plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x='white_rating', y='black_rating', hue='Cluster', data=df, palette='Set1', s=50, ax=ax)
    ax.set_title(f'Agglomerative Clustering (Linkage: {linkage_type})')
    ax.set_xlabel('White Player Rating')
    ax.set_ylabel('Black Player Rating')
    st.pyplot(fig)

    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

    # Display clustering metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Silhouette Score", value=f"{silhouette_avg:.2f}")

    with col2:
        st.metric(label="Davies-Bouldin Index", value=f"{dbi:.2f}")

    with col3:
        st.metric(label="Calinski-Harabasz Index", value=f"{calinski_harabasz:.2f}")

    # Analyze the size of each cluster
    cluster_sizes = df['Cluster'].value_counts()
    st.write("Cluster Sizes")
    st.write(cluster_sizes)

        # Range of potential clusters
    range_n_clusters = list(range(2, 10))

    # Silhouette scores for different numbers of clusters
    silhouette_avg = []
    for n_clusters in range_n_clusters:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clusterer.fit_predict(X_scaled)
        silhouette_avg.append(silhouette_score(X_scaled, cluster_labels))

    # Plotting silhouette scores
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.plot(range_n_clusters, silhouette_avg, marker='o')
    ax3.set_title('Silhouette Score for Various Numbers of Clusters')
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Average Silhouette Score')
    ax3.grid(True)
    st.pyplot(fig3)

    # Davies-Bouldin Index for different numbers of clusters
    db_scores = []
    for i in range(2, 10):
        model = AgglomerativeClustering(n_clusters=i, metric='euclidean', linkage='ward')
        labels = model.fit_predict(X_scaled)
        score = davies_bouldin_score(X_scaled, labels)
        db_scores.append(score)

    # Plotting Davies-Bouldin Index
    fig4, ax4 = plt.subplots(figsize=(4, 3))
    ax4.plot(range(2, 10), db_scores)
    ax4.set_xlabel('Number of Clusters')
    ax4.set_ylabel('Davies-Bouldin Index')
    ax4.set_title('Davies-Bouldin Index Method')
    st.pyplot(fig4)

    # Calinski-Harabasz Index for different numbers of clusters
    ch_scores = []
    for i in range(2, 10):
        model = AgglomerativeClustering(n_clusters=i, metric='euclidean', linkage='ward')
        labels = model.fit_predict(X_scaled)
        score = calinski_harabasz_score(X_scaled, labels)
        ch_scores.append(score)

    # Plotting Calinski-Harabasz Index
    fig5, ax5 = plt.subplots(figsize=(4, 3))
    ax5.plot(range(2, 10), ch_scores)
    ax5.set_xlabel('Number of Clusters')
    ax5.set_ylabel('Calinski-Harabasz Index')
    ax5.set_title('Calinski-Harabasz Index Method')
    st.pyplot(fig5)

    # Boxplot for white_rating across clusters
    st.write("") 
    st.subheader("Cluster Summary Statistics")

    # Plot the distributions of player ratings for each cluster
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    sns.histplot(data=df, x='white_rating', hue='Cluster', multiple='stack', bins=50, ax=ax1)
    ax1.set_title('White Player Rating Distribution by Cluster')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    sns.histplot(data=df, x='black_rating', hue='Cluster', multiple='stack', bins=50, ax=ax2)
    ax2.set_title('Black Player Rating Distribution by Cluster')
    st.pyplot(fig2)

elif model_type == "BIRCH Clustering":
    n_clusters = 2  # Fixed to 2 clusters as per the goal
    threshold = st.sidebar.slider("Threshold", min_value=0.4, max_value=0.7, value=0.5, step=0.05)
    branching_factor = st.sidebar.slider("Branching Factor", min_value=10, max_value=80, value=30, step=10)

    # Perform BIRCH clustering
    from sklearn.cluster import Birch
    birch_clustering = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
    labels = birch_clustering.fit_predict(X_scaled)

    # Add the cluster labels to your DataFrame
    df['Cluster'] = labels

    # Visualize the clusters using a scatter plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x='white_rating', y='black_rating', hue='Cluster', data=df, palette='Set1', s=50, ax=ax)
    ax.set_title(f'BIRCH Clustering (Threshold: {threshold}, Branching Factor: {branching_factor})')
    ax.set_xlabel('White Player Rating')
    ax.set_ylabel('Black Player Rating')
    st.pyplot(fig)

    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

    # Display clustering metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Silhouette Score", value=f"{silhouette_avg:.2f}")

    with col2:
        st.metric(label="Davies-Bouldin Index", value=f"{dbi:.2f}")

    with col3:
        st.metric(label="Calinski-Harabasz Index", value=f"{calinski_harabasz:.2f}")

    # Analyze the size of each cluster
    cluster_sizes = df['Cluster'].value_counts()
    st.write("Cluster Sizes")
    st.write(cluster_sizes)

    # Range of potential clusters for evaluation
    range_n_clusters = list(range(2, 10))

    # Metric scores for each cluster count
    silhouette_scores = []
    dbi_scores = []
    calinski_harabasz_scores = []

    # Evaluate different numbers of clusters
    for n_clusters in range_n_clusters:
        birch_temp = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
        labels_temp = birch_temp.fit_predict(X_scaled)

        # Compute metrics
        silhouette_scores.append(silhouette_score(X_scaled, labels_temp))
        dbi_scores.append(davies_bouldin_score(X_scaled, labels_temp))
        calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, labels_temp))

    # Plot Silhouette Scores
    fig_silhouette, ax_silhouette = plt.subplots(figsize=(4, 3))
    ax_silhouette.plot(range_n_clusters, silhouette_scores, marker='o')
    ax_silhouette.set_title('Silhouette Score for Various Numbers of Clusters')
    ax_silhouette.set_xlabel('Number of Clusters')
    ax_silhouette.set_ylabel('Silhouette Score')
    st.pyplot(fig_silhouette)

    # Plot Davies-Bouldin Index
    fig_dbi, ax_dbi = plt.subplots(figsize=(4, 3))
    ax_dbi.plot(range_n_clusters, dbi_scores, marker='o')
    ax_dbi.set_title('Davies-Bouldin Index for Various Numbers of Clusters')
    ax_dbi.set_xlabel('Number of Clusters')
    ax_dbi.set_ylabel('Davies-Bouldin Index')
    st.pyplot(fig_dbi)

    # Plot Calinski-Harabasz Index
    fig_calinski, ax_calinski = plt.subplots(figsize=(4, 3))
    ax_calinski.plot(range_n_clusters, calinski_harabasz_scores, marker='o')
    ax_calinski.set_title('Calinski-Harabasz Index for Various Numbers of Clusters')
    ax_calinski.set_xlabel('Number of Clusters')
    ax_calinski.set_ylabel('Calinski-Harabasz Index')
    st.pyplot(fig_calinski)

    # Boxplot for white_rating across clusters
    st.write("")
    st.subheader("Cluster Summary Statistics")

    # Boxplot for white_rating across clusters
    fig_white, ax_white = plt.subplots()
    sns.boxplot(x='Cluster', y='white_rating', data=df, ax=ax_white)
    ax_white.set_title('White Rating Distribution Across Clusters')
    st.pyplot(fig_white)

    # Boxplot for black_rating across clusters
    fig_black, ax_black = plt.subplots()
    sns.boxplot(x='Cluster', y='black_rating', data=df, ax=ax_black)
    ax_black.set_title('Black Rating Distribution Across Clusters')
    st.pyplot(fig_black)

elif model_type == "Gaussian Mixture Model (GMM)":
    st.write("")
    st.subheader("Gaussian Mixture Model")
    # Sidebar for GMM parameters
    covariance_type = st.sidebar.selectbox("Select Covariance Type", ['full', 'tied', 'diag', 'spherical'])
    n_components = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=10)

    # Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    # Add the cluster labels to your DataFrame
    df['Cluster'] = labels

    # Visualize the clusters using a scatter plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x='white_rating', y='black_rating', hue='Cluster', data=df, palette='Set1', s=50, ax=ax)
    ax.set_title(f'GMM Clustering (Covariance Type: {covariance_type}, k={n_components})')
    ax.set_xlabel('White Player Rating')
    ax.set_ylabel('Black Player Rating')
    st.pyplot(fig)

    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)

    # Display clustering metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Silhouette Score", value=f"{silhouette_avg:.2f}")

    with col2:
        st.metric(label="Davies-Bouldin Index", value=f"{dbi:.2f}")

    # Evaluate clustering metrics over a range of cluster counts
    range_n_clusters = list(range(2, 10))
    silhouette_scores = []
    dbi_scores = []

    for n_components in range_n_clusters:
        gmm_temp = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
        labels_temp = gmm_temp.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels_temp))
        dbi_scores.append(davies_bouldin_score(X_scaled, labels_temp))

    # Plot Silhouette Scores
    st.write("")
    st.subheader("Silhoutte Scores")
    fig_silhouette, ax_silhouette = plt.subplots(figsize=(4, 3))
    ax_silhouette.plot(range_n_clusters, silhouette_scores, marker='o')
    ax_silhouette.set_title('Silhouette Score for Various Numbers of Clusters (GMM)')
    ax_silhouette.set_xlabel('Number of Clusters')
    ax_silhouette.set_ylabel('Silhouette Score')
    st.pyplot(fig_silhouette)

    # Plot Davies-Bouldin Index
    st.subheader("Davies-Bouldin Index")
    fig_dbi, ax_dbi = plt.subplots(figsize=(4, 3))
    ax_dbi.plot(range_n_clusters, dbi_scores, marker='o')
    ax_dbi.set_title('Davies-Bouldin Index for Various Numbers of Clusters (GMM)')
    ax_dbi.set_xlabel('Number of Clusters')
    ax_dbi.set_ylabel('Davies-Bouldin Index')
    st.pyplot(fig_dbi)

    # Boxplot for white_rating across clusters
    st.write("")
    st.subheader("Cluster Summary Statistics (GMM)")

    # Boxplot for white_rating across clusters (GMM)
    fig_white_gmm, ax_white_gmm = plt.subplots()
    sns.boxplot(x='Cluster', y='white_rating', data=df, ax=ax_white_gmm)
    ax_white_gmm.set_title('White Rating Distribution Across Clusters (GMM)')
    st.pyplot(fig_white_gmm)

    # Boxplot for black_rating across clusters (GMM)
    fig_black_gmm, ax_black_gmm = plt.subplots()
    sns.boxplot(x='Cluster', y='black_rating', data=df, ax=ax_black_gmm)
    ax_black_gmm.set_title('Black Rating Distribution Across Clusters (GMM)')
    st.pyplot(fig_black_gmm)

elif model_type == "K Means Clustering":
    st.write("")
    st.subheader("K Means Clustering")
    # Sidebar for K-Means parameters
    n_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=2)

    # K-Means Clustering with user-defined number of clusters
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.predict(X_scaled)

    # Add cluster labels to the DataFrame
    df['cluster'] = labels

    # Compute clustering metrics
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)

    # Visualize the clusters using a scatter plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x='white_rating', y='black_rating', hue='cluster', data=df, palette='viridis', s=100, ax=ax)
    ax.set_title(f'K-Means Clustering ({n_clusters} Clusters)')
    ax.set_xlabel('White Rating')
    ax.set_ylabel('Black Rating')
    st.pyplot(fig)

    # Display clustering metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Silhouette Score", value=f"{silhouette:.2f}")

    with col2:
        st.metric(label="Davies-Bouldin Index", value=f"{davies_bouldin:.2f}")

    # Evaluate clustering metrics over a range of cluster counts
    range_n_clusters = list(range(2, 10))
    silhouette_scores = []
    dbi_scores = []

    for n_clusters in range_n_clusters:
        kmeans_temp = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels_temp = kmeans_temp.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels_temp))
        dbi_scores.append(davies_bouldin_score(X_scaled, labels_temp))

    # Plot Silhouette Scores
    st.write("")
    st.subheader("Silhouette Scores")
    fig_silhouette, ax_silhouette = plt.subplots(figsize=(4, 3))
    ax_silhouette.plot(range_n_clusters, silhouette_scores, marker='o')
    ax_silhouette.set_title('Silhouette Score for Various Numbers of Clusters (K-Means)')
    ax_silhouette.set_xlabel('Number of Clusters')
    ax_silhouette.set_ylabel('Silhouette Score')
    st.pyplot(fig_silhouette)

    # Plot Davies-Bouldin Index
    st.subheader("Davies-Bouldin Index")
    fig_dbi, ax_dbi = plt.subplots(figsize=(4, 3))
    ax_dbi.plot(range_n_clusters, dbi_scores, marker='o')
    ax_dbi.set_title('Davies-Bouldin Index for Various Numbers of Clusters (K-Means)')
    ax_dbi.set_xlabel('Number of Clusters')
    ax_dbi.set_ylabel('Davies-Bouldin Index')
    st.pyplot(fig_dbi)

    # Boxplot for white_rating across clusters
    st.write("")
    st.subheader("Cluster Summary Statistics (K-Means)")

    # Boxplot for white_rating across clusters (K-Means)
    fig_white_kmeans, ax_white_kmeans = plt.subplots()
    sns.boxplot(x='cluster', y='white_rating', data=df, ax=ax_white_kmeans)
    ax_white_kmeans.set_title('White Rating Distribution Across Clusters (K-Means)')
    st.pyplot(fig_white_kmeans)

    # Boxplot for black_rating across clusters (K-Means)
    fig_black_kmeans, ax_black_kmeans = plt.subplots()
    sns.boxplot(x='cluster', y='black_rating', data=df, ax=ax_black_kmeans)
    ax_black_kmeans.set_title('Black Rating Distribution Across Clusters (K-Means)')
    st.pyplot(fig_black_kmeans)