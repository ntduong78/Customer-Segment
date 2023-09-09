import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import time
from sklearn.metrics import silhouette_score
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans as py_KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.functions import avg, count
from pyspark.ml.evaluation import ClusteringEvaluator
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import joblib 
import plotly.io as pio
import pickle
from PIL import Image
import os
from datetime import datetime
import joblib
import re
pio.templates.default = 'plotly' 

# Define the global current_date variable
current_date = datetime.now()

# Define a function to load data
def load_data(data_path):
    
    columns = ['customer_id', 'transaction_date', 'num_cds_purchased', 'transaction_value']
    df = pd.read_csv(data_path, sep='\s+', header=None, names=columns)
    return df

def explore_prepare_data(df):
    # Display sample data after loading
    st.write("##### Sample data")
    st.dataframe(df.sample(5))

    # Summary statistics
    st.write("##### Summary Statistics")
    st.write(df.describe())
        
    # Check for missing values and return as a DataFrame
    st.write("##### Missing values")
    missing_values_df = pd.DataFrame(df.isna().sum(), columns=['Missing_Values'])
    st.dataframe(missing_values_df)

    # Convert 'transaction_date' to datetime64 format
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%Y%m%d')

    # Remove rows with negative or zero transaction values
    df = df[df['transaction_value'] > 0]

    # Drop NA values
    df = df.dropna()

    # Additional Information
    st.write("##### Additional Information")
    current_date = pd.to_datetime('today')  # Assuming you want to calculate this for the current date
    st.write("Transactions timeframe from {} to {}".format(df['transaction_date'].min(), df['transaction_date'].max()))
    st.write("{:,} transactions don't have a customer id".format(df[df.customer_id.isnull()].shape[0]))
    st.write("{:,} unique customer_id".format(len(df.customer_id.unique())))

    # Create RFM analysis for each customers
    current_date = df['transaction_date'].max()

    df_RFM = df.groupby('customer_id').agg({
        'transaction_date': lambda x: (current_date - x.max()).days,
        'transaction_value': ['count', 'sum']
    })

    df_RFM.columns = ['Recency', 'Frequency', 'Monetary']

    return df_RFM

def explore_data(df):
    # Display sample data after loading
    st.write("##### Sample data")
    st.dataframe(df.sample(5))

    # Summary statistics
    st.write("##### Summary Statistics")
    st.write(df.describe())

    # Check for missing values and return as a DataFrame
    st.write("##### Missing values")
    missing_values_df = pd.DataFrame(df.isna().sum(), columns=['Missing_Values'])
    st.dataframe(missing_values_df)

    # Additional Information
    st.write("##### Additional Information")
    current_date = pd.to_datetime('today')  # Assuming you want to calculate this for the current date
    st.write("Transactions timeframe from {} to {}".format(df['transaction_date'].min(), df['transaction_date'].max()))
    st.write("{:,} transactions don't have a customer id".format(df[df.customer_id.isnull()].shape[0]))
    st.write("{:,} unique customer_id".format(len(df.customer_id.unique())))

def prepare_data(df):
    # Convert 'transaction_date' to datetime64 format
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%Y%m%d')

    # Remove rows with negative or zero transaction values
    df = df[df['transaction_value'] > 0]

    # Drop NA values
    df = df.dropna()

    # Create RFM analysis for each customers
    current_date = df['transaction_date'].max()

    df_RFM = df.groupby('customer_id').agg({
        'transaction_date': lambda x: (current_date - x.max()).days,
        'transaction_value': ['count', 'sum']
    })

    df_RFM.columns = ['Recency', 'Frequency', 'Monetary']

    return df_RFM, current_date

def rfm_level2(df):
    if (df['RFM_score'] == 12):
        return 'VIP Customers'
    elif (df['R'] >= 2 and df['F'] == 4):
        return 'Potential Loyalists'
    elif (df['F'] <= 2 and df['M'] <= 2 and df['R'] <= 2):
        return 'Lost'
    else:
        return 'Regular'
    
def rfm_level1(df):
    if (df['RFM_score'] == 12):
        return 'VIP Customers'
    elif (df['R'] >= 3):
        return 'Active Customers'
    else:
        return 'Lost Customers'
    
def rfm_level(df):
    if (df['RFM_score'] >= 9):
        return 'Active Customers'
    else:
        return 'Lost Customers'
        
# Define a function for RFM analysis
def perform_rfm_analysis(df_RFM,num_clusters=2):
    
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)

    r_groups = pd.qcut(df_RFM['Recency'].rank(method='first'), q=4, labels=r_labels)
    f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=4, labels=f_labels)
    m_groups = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=4, labels=m_labels)

    df_RFM = df_RFM.assign(R=r_groups.values, F=f_groups.values, M=m_groups.values)

    def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
    df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)

    fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    sns.histplot(data=df_RFM, x='Recency', bins=30, kde=True, ax=ax1)
    ax1.set_title('Recency Distribution')
    ax1.set_xlabel('Recency')
    ax1.set_ylabel('Density')

    sns.histplot(data=df_RFM, x='Frequency', bins=30, kde=True, ax=ax2)
    ax2.set_title('Frequency Distribution')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Density')

    sns.histplot(data=df_RFM, x='Monetary', bins=30, kde=True, ax=ax3)
    ax3.set_title('Monetary Distribution')
    ax3.set_xlabel('Monetary')
    ax3.set_ylabel('Density')

    plt.tight_layout()
    fig1.savefig("RFM_Distribution.png")  # Save the figure
    st.write("##### RFM Distribution")
    st.image("RFM_Distribution.png")  # Display the saved image

    # Comments for the charts
    # Recency Distribution Comment
    st.markdown("**Recency Distribution:**")
    st.markdown("- The majority of customers have made purchases in the year before.")
    st.markdown("- There is a gradual decline in the number of customers as recency increases, suggesting that the customer base becomes less active over time.")

    # Frequency Distribution Comment
    st.markdown("**Frequency Distribution:**")
    st.markdown("- The frequency distribution is skewed to the right, indicating that a significant portion of customers make fewer purchases (less than 10 times).")

    # Monetary Distribution Comment
    st.markdown("**Monetary Distribution:**")
    st.markdown("- Similar to the frequency distribution, the monetary distribution is also right-skewed, with most customers making lower-value transactions (less than 500).")

    rfm_count_unique = df_RFM.groupby('RFM_Segment')['RFM_Segment'].nunique()
    st.write("Number of unique segment before manual segmentation is: ", rfm_count_unique.sum())
    st.markdown("- Having 47 different segments using the concatenate method quickly becomes unwieldy for any practical use. We will need a more concise way to define our segments.")

    start_time = time.time()
    df_RFM['RFM_score'] = df_RFM[['R', 'F', 'M']].sum(axis=1)
    if num_clusters == 2:
        df_RFM['RFM_level'] = df_RFM.apply(rfm_level, axis=1)
    else:
        df_RFM['RFM_level'] = df_RFM.apply(rfm_level1, axis=1)

    rfm_agg = df_RFM.groupby('RFM_level').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(0)

    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
    rfm_agg['Percent'] = round((rfm_agg['Count'] / rfm_agg.Count.sum()) * 100, 2)

    rfm_agg = rfm_agg.reset_index()

    rfm_results = rfm_agg.sort_values(by='Percent', ascending=False)
    time_taken = time.time() - start_time

    st.write("##### RFM Customer Segmentation Results")
    st.dataframe(rfm_results)
    st.write("##### Time taken to build RFM manually: ", round(time_taken, 2))

    fig2 = plt.figure(figsize=(14, 10))
    ax_2 = fig2.add_subplot()

    colors_dict = {'ACTIVE': 'yellow', 'BIG SPENDER': 'royalblue', 'LIGHT': 'cyan',
                   'LOST': 'red', 'LOYAL': 'purple', 'POTENTIAL': 'green', 'STARS': 'gold', 'ONE-TIME': 'orange'}

    squarify.plot(sizes=rfm_agg['Count'],
                  text_kwargs={'fontsize': 12, 'weight': 'bold', 'fontname': "sans serif"},
                  color=colors_dict.values(),
                  label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                         for i in range(0, len(rfm_agg))], alpha=0.5)

    plt.title("RFM Customer Segments Tree Map", fontsize=26, fontweight="bold")
    plt.axis('off')
    
    fig2.savefig('rfm_tree_map.png',bbox_inches='tight')  # Save the figure
    st.write("##### RFM Customer Segments Tree Map")
    st.image("rfm_tree_map.png")  # Display the saved image

     # Create a color scale (e.g., viridis, rainbow, jet, etc.)
    color_scale = px.colors.sequential.Viridis  # You can change this to your preferred color scale

    fig3 = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_level",
                      hover_name="RFM_level", size_max=100, color_continuous_scale=color_scale)
    
    # Save the scatter plot as an image file using Plotly's to_image method
    scatter_image = pio.to_image(fig3, format="png", width=800, height=600, scale=1)

    #fig3.write_image("RFM_Scatter.png")  # Save the scatter plot as an image
    st.write("##### RFM Customer Segments Scatter Plot")
    # Save the image to a file
    scatter_image_file = "rfm_scatter_plot.png"
    with open(scatter_image_file, "wb") as img_file:
        img_file.write(scatter_image)

    st.image(scatter_image_file)

    return rfm_results, time_taken


def plot_elbow_chart(data, max_clusters=20):
    sse = {}
    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

    # Create the Elbow Method chart
    fig0, ax0 = plt.subplots()
    ax0.set_title('The Elbow Method')
    ax0.set_xlabel('k')
    ax0.set_ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()), ax=ax0)
    
    st.pyplot(fig0)

# Define a function to build the K-Means model with the specified k value
def build_kmeans_model(data, k):
    """
    Build a K-Means clustering model.

    Parameters:
        data (array-like): The data to cluster.
        k (int): The number of clusters to create.

    Returns:
        kmeans: The trained K-Means model.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(data)
    
    # Save the K-Means model to a file
    joblib.dump(kmeans, 'kmeans_model.pkl')

    return kmeans

# Define a function for K-Means clustering
def perform_kmeans_clustering(df_RFM, num_clusters=2):
    """
    Perform K-Means clustering on RFM (Recency, Frequency, Monetary) data and display results.

    Parameters:
        df_RFM (pd.DataFrame): DataFrame containing RFM data.
        num_clusters (int): Number of clusters to create using K-Means (default is 4).

    Returns:
        pd.DataFrame: DataFrame with cluster assignments and aggregated statistics.
        float: Time taken to build the K-Means model.
        float: Silhouette score for the clustering.
    """

    # Select the columns to use for clustering
    rfm_selected = df_RFM[['Recency', 'Frequency', 'Monetary']]

    # Scale the features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_selected)

    st.write("##### Elbow Method for K-Means Clustering")
    plot_elbow_chart(rfm_scaled)
    st.markdown(f"**Select k = {num_clusters}**")

    start_time = time.time()  # Start timing the function

    # Build the K-Means model with the specified number of clusters
    kmeans = build_kmeans_model(rfm_scaled, num_clusters)

    # End timing and calculate the time taken
    end_time = time.time()
    time_taken = end_time - start_time

    # Create a new DataFrame with the original data and cluster assignments
    clustered_data = pd.DataFrame({'customer_id': df_RFM.index, 'Cluster': kmeans.labels_})

    # Merge with the original data
    result_data = pd.merge(rfm_selected, clustered_data, left_index=True, right_on='customer_id')
    # Write result clustering to csv
    result_data.to_csv('result_data.csv', index=False)


    # Calculate average values for each cluster
    rfm_agg2 = result_data.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg2.columns = rfm_agg2.columns.droplevel()
    rfm_agg2.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
    rfm_agg2['Percent'] = round((rfm_agg2['Count'] / rfm_agg2.Count.sum()) * 100, 2)

    # Reset the index
    rfm_agg2 = rfm_agg2.reset_index()

    # Change the Cluster column's datatype into discrete values
    rfm_agg2['Cluster'] = 'Cluster ' + rfm_agg2['Cluster'].astype('str')
    st.write("##### KMeans Customer Segmentation Results")
    st.dataframe(rfm_agg2)

    st.write("##### Time taken to build KMean model: ", round(time_taken, 2))

    # Calculate silhouette score for evaluation
    silhouette_avg = silhouette_score(rfm_scaled, kmeans.labels_)
    st.write("##### Silhouette score: ", round(silhouette_avg, 2))

    # Visualization KMeans
    # Create a separate figure for the tree map
    fig1, ax_1 = plt.subplots(figsize=(14, 10))

    colors_dict2 = {'Cluster0': 'yellow', 'Cluster1': 'royalblue', 'Cluster2': 'cyan',
                    'Cluster3': 'red', 'Cluster4': 'purple', 'Cluster5': 'green', 'Cluster6': 'gold'}

    squarify.plot(sizes=rfm_agg2['Count'],
                  text_kwargs={'fontsize': 12, 'weight': 'bold', 'fontname': "sans serif"},
                  color=colors_dict2.values(),
                  label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2.iloc[i])
                         for i in range(0, len(rfm_agg2))], alpha=0.5)

    plt.title("KMeans Customers Segments Tree Map", fontsize=26, fontweight="bold")
    plt.axis('off')

    # Save the tree map as an image file
    fig1.savefig("kmeans_tree_map.png", bbox_inches='tight')

    # Display the saved tree map using st.image
    st.image("kmeans_tree_map.png")

    # Create a color scale (e.g., viridis, rainbow, jet, etc.)
    color_scale = px.colors.sequential.Viridis  # You can change this to your preferred color scale

    # Scatter plot with the chosen color scale
    fig2 = px.scatter(rfm_agg2, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
                    hover_name="Cluster", size_max=100, color_continuous_scale=color_scale)

    # Save the scatter plot as an image file using Plotly's to_image method
    scatter_image = pio.to_image(fig2, format="png", width=800, height=600, scale=1)

    # Save the image to a file
    with open("kmeans_scatter_plot.png", "wb") as img_file:
        img_file.write(scatter_image)

    st.write("##### KMeans Customer Segments Scatter Plot")
    # Display the saved scatter plot using st.image
    st.image("kmeans_scatter_plot.png")

    return rfm_agg2, time_taken, silhouette_avg

def plot_elbow_chart_pyspark(data, featuresCol, max_clusters=20):
    # Calculate inertia for different values of k
    inertia = []
    for k in range(2, max_clusters):  # Starting from k = 2
        kmeans = py_KMeans(k=k, seed=42, featuresCol=featuresCol)
        kmeans_model = kmeans.fit(data)
        inertia.append(kmeans_model.summary.trainingCost)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 20), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.xticks(np.arange(2, 20, step=1))
    plt.grid(True)
    st.pyplot(plt)

# Define a function to build the K-Means model with the specified k value
def build_kmeans_model_pyspark(data, k, featuresCol):
    model = py_KMeans(n_clusters=k, seed=42, featuresCol=featuresCol)
    cluster_assignments = model.fit_predict(data)
    return cluster_assignments

# Define function for KMean pyspark clustering
def perform_pyspark_kmeans_clustering(df_RFM, num_clusters=4):
    # Create a SparkSession
    spark = SparkSession.builder.appName("CustomerSegmentation").getOrCreate()

    # Create a DataFrame from pandas DataFrame
    spark_df = spark.createDataFrame(df_RFM)

    # Create a VectorAssembler to assemble features
    feature_columns = ["Recency", "Frequency", "Monetary"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    spark_df = assembler.transform(spark_df)

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

    # Fit and transform the scaler
    scaler_model = scaler.fit(spark_df)
    scaled_df = scaler_model.transform(spark_df)

    # Elbow method to select k
    plot_elbow_chart_pyspark(scaled_df, "scaled_features")

    start_time = time.time()  # Start timing the function
    kmeans_model = build_kmeans_model_pyspark(scaled_df, num_clusters, "scaled_features")
    
    # Calculate execution time
    execution_time = time.time() - start_time

    # Add cluster labels to the DataFrame
    clustered_df = kmeans_model.transform(scaled_df)

    # Create a ClusteringEvaluator
    evaluator = ClusteringEvaluator()

    # Calculate silhouette score using the evaluator
    silhouette = evaluator.evaluate(clustered_df)

    # Display results
    result_data = clustered_df.toPandas()

    # Calculate average values for each cluster
    rfm_agg2 = result_data.groupby("prediction").agg(
        avg("Recency").alias("RecencyMean"),
        avg("Frequency").alias("FrequencyMean"),
        avg("Monetary").alias("MonetaryMean"),
        count("Monetary").alias("Count")  # Calculate count using count function
    )

    # Calculate the percentage for each cluster
    total_count = result_data.shape[0]
    rfm_agg2 = rfm_agg2.withColumn("Percent", (rfm_agg2["Count"] / total_count) * 100)

    # Convert DataFrame to Pandas
    rfm_agg2_pandas = rfm_agg2.toPandas()

    # Add Cluster prefix to Cluster column
    rfm_agg2_pandas['prediction'] = 'Cluster ' + rfm_agg2_pandas['prediction'].astype(str)

    # Sort rfm_agg2_pandas by the 'prediction' column in descending order
    rfm_agg2_pandas_sorted = rfm_agg2_pandas.sort_values(by='prediction', ascending=False)

    # Round numerical columns to 2 decimal places
    rfm_agg2_pandas_sorted = rfm_agg2_pandas_sorted.round(2)

    # Save the DataFrame as a CSV file
    rfm_agg2_pandas_sorted.to_csv("rfm_agg2_pandas_sorted.csv", index=False)

    st.dataframe(rfm_agg2_pandas_sorted)

    st.write("##### Time taken to build Pyspark KMean model: ",round(execution_time,2))
    st.write("##### Silhouette score: ", round(silhouette,2))

    # Create a figure for the tree map
    fig1, ax1 = plt.subplots(figsize=(14, 10))

    colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
                    'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

    squarify.plot(sizes=rfm_agg2_pandas_sorted['Count'],
                  text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                  color=colors_dict2.values(),
                  label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2_pandas_sorted.iloc[i])
                          for i in range(0, len(rfm_agg2_pandas_sorted))], alpha=0.5 )

    plt.title("Pyspark Kmean Customers Segments Tree Map", fontsize=26, fontweight="bold")
    plt.axis('off')

    # Save the tree map as an image file
    fig1.savefig("py_kmeans_tree_map.png", bbox_inches='tight')

    # Display the saved image using st.image
    st.image("py_kmeans_tree_map.png")

    # Create a color scale (e.g., viridis, rainbow, jet, etc.)
    color_scale = px.colors.sequential.Viridis  # You can change this to your preferred color scale

    # Create a scatter plot
    fig2 = px.scatter(rfm_agg2_pandas_sorted, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="prediction",
               hover_name="prediction", size_max=100, color_continuous_scale=color_scale)

    # Display the saved image using st.image
    st.write("### Scatter Plot of Customer Segments")

    # Save the scatter plot as an image file using Plotly's to_image method
    scatter_image = pio.to_image(scatter_fig, format="png", width=800, height=600, scale=1)

    # Save the image to a file
    scatter_image_file = "hierarchical_scatter_plot.png"
    with open(scatter_image_file, "wb") as img_file:
        img_file.write(scatter_image)

    st.image(scatter_image_file)

    # Save the KMeans model
    kmeans_model.save("pyspark_kmeans_model")

    return rfm_agg2_pandas_sorted, execution_time, silhouette


# Define function to perform hierachical clustering
def perform_hierarchical_clustering(df_RFM, num_clusters=2):
    # Scale the features
    rfm_selected = df_RFM[['Recency', 'Frequency', 'Monetary']]
    rfm_scaled = StandardScaler().fit_transform(rfm_selected)

    # Perform hierarchical clustering
    #linked = linkage(rfm_scaled, method='ward')

    # Create a Streamlit figure to display the dendrogram
    #dendrogram_fig = plt.figure(figsize=(12, 6))
    #dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    #plt.title('Dendrogram of Customer Segments')
    #plt.xlabel('Customer ID')
    #plt.ylabel('Distance')

    # Save the dendrogram as an image file
    #dendrogram_fig.savefig("dendrogram_image.png", bbox_inches='tight')
    
    st.write("###### Dendrogram of Customer Segments")
    st.image("dendrogram_image.png")
    
    # Fit Hierarchical Clustering model
    start_time = time.time()  # Start timing the model fitting
    agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    cluster_assignments = agg_clustering.fit_predict(rfm_scaled)
    end_time = time.time()  # End timing the model fitting

    execution_time = end_time - start_time  # Calculate execution time

    # Calculate silhouette score
    silhouette = silhouette_score(rfm_scaled, cluster_assignments)

    # Create a new DataFrame with the original data and cluster assignments
    clustered_data2 = pd.DataFrame({'customer_id': df_RFM.index, 'Cluster': cluster_assignments})

    # Merge with the original data
    result_data2 = pd.merge(rfm_selected, clustered_data2, left_index=True, right_on='customer_id')

    # Calculate average values for each RFM_Level, and return a size of each segment
    rfm_agg3 = result_data2.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg3.columns = rfm_agg3.columns.droplevel()
    rfm_agg3.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
    rfm_agg3['Percent'] = round((rfm_agg3['Count'] / rfm_agg3.Count.sum()) * 100, 2)

    # Reset the index
    rfm_agg3 = rfm_agg3.reset_index()

    # Change the Cluster column's datatype into discrete values
    rfm_agg3['Cluster'] = 'Cluster ' + rfm_agg3['Cluster'].astype('str')

    # Display the aggregated dataset
    st.write("###### Aggregated Customer Segments")
    st.dataframe(rfm_agg3)

    # Display execution time and silhouette score
    st.write("###### Time taken to build Hierarchical Clustering model: ", round(execution_time, 2))
    st.write("###### Silhouette score: ", round(silhouette, 2))

    # Create a figure for the tree map
    squarify_fig = plt.figure(figsize=(14, 10))
    colors_dict2 = {'Cluster0': 'yellow', 'Cluster1': 'royalblue', 'Cluster2': 'cyan',
                    'Cluster3': 'red', 'Cluster4': 'purple', 'Cluster5': 'green', 'Cluster6': 'gold'}
    squarify.plot(sizes=rfm_agg3['Count'],
                  text_kwargs={'fontsize': 12, 'weight': 'bold', 'fontname': "sans serif"},
                  color=colors_dict2.values(),
                  label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg3.iloc[i])
                          for i in range(0, len(rfm_agg3))], alpha=0.5)
    plt.title("Hierachical Clustering Customer Segments", fontsize=26, fontweight="bold")
    plt.axis('off')

    # Save the tree map as an image file
    squarify_fig.savefig("hierarchical_tree_map.png", bbox_inches='tight')
    st.write("###### Hierachical Clustering Tree Map")
    st.image("hierarchical_tree_map.png")

    # Create a color scale (e.g., viridis, rainbow, jet, etc.)
    color_scale = px.colors.sequential.Viridis  # You can change this to your preferred color scale

    # Scatter plot with the chosen color scale
    scatter_fig = px.scatter(rfm_agg3, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
                            hover_name="Cluster", size_max=100, color_continuous_scale=color_scale)

    st.write("###### Hierachical Clustering Scatter Plot")

    # Save the scatter plot as an image file using Plotly's to_image method
    scatter_image = pio.to_image(scatter_fig, format="png", width=800, height=600, scale=1)

    # Save the image to a file
    scatter_image_file = "hierarchical_scatter_plot.png"
    with open(scatter_image_file, "wb") as img_file:
        img_file.write(scatter_image)

    st.image(scatter_image_file)

    # Save the Hierarchical Clustering model to a file
    with open("hierarchical_clustering_model.pkl", "wb") as model_file:
        pickle.dump(agg_clustering, model_file)

    # Return results
    return rfm_agg3, execution_time, silhouette

# Define a function to update or add records in global_results_df

def update_global_results(model_name, execution_time, silhouette_score, csv_filename='comparison.csv'):
    # Load the existing comparison data from the CSV file
    try:
        global_results_df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty DataFrame
        global_results_df = pd.DataFrame(columns=["Model", "Execution Time (s)", "Silhouette Score"])

    # Check if the model name already exists in the DataFrame
    existing_index = global_results_df[global_results_df["Model"] == model_name].index

    if not existing_index.empty:
        # Replace the existing record with the same model name
        global_results_df.loc[existing_index, ["Execution Time (s)", "Silhouette Score"]] = [execution_time, silhouette_score]
    else:
        # Add a new record for the model
        new_record = {"Model": model_name, "Execution Time (s)": execution_time, "Silhouette Score": silhouette_score}
        global_results_df = pd.concat([global_results_df, pd.DataFrame([new_record])], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    global_results_df.to_csv(csv_filename, index=False)

# Create a global DataFrame to store results
global_results_df = pd.DataFrame(columns=["Model", "Execution Time (s)", "Silhouette Score"])


# ---------------------------------
# GUI
# Create a Streamlit app
st.title("Customer Segmentation Project")

# Sidebar menu options
menu = ["Business Objective", "Build Project", "Model Selection", "New Prediction"]

# Sidebar selection
selected_option = st.sidebar.selectbox('Menu', menu, key="menu_sidebar")

if selected_option == 'Business Objective':
    st.subheader("Business Objective")
    st.write("Building a customer segmentation system based on the information provided by the company, which can help the company identify different customer segments for business strategies and appropriate customer care.")
    st.write("=> Problem/Requirement: Use RFM and then RFM together with clustering algorithms in Python for customer segmentation.")
    st.image("Customer_Segmentation.png")

elif selected_option == 'Build Project':
    data_path = 'CDNOW_master/CDNOW_master.txt'
    df = load_data(data_path)
    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    #st.write(uploaded_file.name)
    if uploaded_file is not None:
        file_name = uploaded_file.name
        # Load and preprocess the data
        df = load_data(file_name)
        df.to_csv("CDNOW_master_new.txt", index=False)
    st.write(df.shape)
    # Create a dropdown or radio button widget to select the number of clusters
    num_clusters = st.radio("Select Number of Clusters", [2, 3])

    st.subheader("Build Project")

    st.write("### 1. Data Exploration and Prepraration")
    explore_data(df)
    df_RFM, current_date = prepare_data(df)

    st.write("### 2. Build Models")
    
    # Perform RFM analysis
    if st.button("Perform RFM Analysis",key="rfm"):
        rfm_results, rfm_time = perform_rfm_analysis(df_RFM,num_clusters)
        
        # Call the function to update or add the record
        update_global_results("RFM Analysis", rfm_time, 0)

        # Define the file path where you want to save the CSV file
        csv_file_path = "rfm_results.csv"

        # Use the to_csv method to write the DataFrame to a CSV file
        rfm_results.to_csv(csv_file_path, index=False)

    # Perform K-Means clustering
    if st.button("Perform K-Means Clustering",key="kmeans"):
        rfm_selected = df_RFM[['Recency','Frequency','Monetary']]
        kmeans_results, kmean_time, kmean_silhouette  = perform_kmeans_clustering(rfm_selected,num_clusters)
        
        update_global_results("KMean Clustering", kmean_time, kmean_silhouette)

        # Define the file path where you want to save the CSV file
        csv_file_path = "kmeans_results.csv"

        # Use the to_csv method to write the DataFrame to a CSV file
        kmeans_results.to_csv(csv_file_path, index=False)

        
    # # Perform Hierachical Clustering
    # if st.button("Perform Hierachical Clustering",key="tree"):
    #     rfm_selected = df_RFM[['Recency','Frequency','Monetary']]
    #     tree_results, tree_time, tree_silhouette = perform_hierarchical_clustering(rfm_selected,num_clusters)
    #     update_global_results("Hierachical Clustering", tree_time, tree_silhouette)
    #     # Define the file path where you want to save the CSV file
    #     csv_file_path = "tree_results.csv"

    #     # Use the to_csv method to write the DataFrame to a CSV file
    #     tree_results.to_csv(csv_file_path, index=False)

    # # Perform K-Means clustering pyspark
    # if st.button("Perform Pyspark K-Means Clustering"):
    #     rfm_selected = df_RFM[['Recency','Frequency','Monetary']]
    #     py_kmeans_results, py_kmean_time, py_kmean_silhouette  = perform_pyspark_kmeans_clustering(rfm_selected)
    #     update_global_results("Pyspark KMeans Clustering", py_kmean_time, py_kmean_silhouette)

elif selected_option == 'Model Selection':
      
    # Check if the "comparison.csv" file exists
    if os.path.isfile('comparison.csv'):
        # Load the comparison data from the CSV file
        comparison_df = pd.read_csv('comparison.csv')

        # Check if there are records in the comparison DataFrame
        if not comparison_df.empty:
            # Display the comparison data
            st.subheader("Model Selection")
            st.write("##### Model Comparison")
            st.dataframe(comparison_df)
        else:
            st.warning("No records found in the comparison data. Please select the 'Build Project' menu to generate results.")
    else:
        st.warning("The 'comparision.csv' file does not exist. Please select the 'Build Project' menu to generate results.")

    st.write("##### Comparision of clustering results")
    # Load individual CSV files and add model name column
    rfm_results = pd.read_csv('rfm_results.csv')
    kmeans_results = pd.read_csv('kmeans_results.csv')
    tree_results = pd.read_csv('tree_results.csv')

    # Rename the 'RFM_Level' column to 'Cluster' in the rfm_results DataFrame
    rfm_results = rfm_results.rename(columns={'RFM_level': 'Cluster'})

    # Add model name columns
    rfm_results['Model'] = 'RFM'
    kmeans_results['Model'] = 'KMeans'
    tree_results['Model'] = 'Hierarchical'

    # Concatenate the DataFrames
    combined_results = pd.concat([rfm_results, kmeans_results, tree_results], ignore_index=True)

    st.dataframe(combined_results)


    # Load the tree map images
    rfm_tree_map = Image.open("rfm_tree_map.png")
    kmeans_tree_map = Image.open("kmeans_tree_map.png")
    hierarchical_tree_map = Image.open("hierarchical_tree_map.png")

    # Load the scatter plot images
    rfm_scatter_plot = Image.open("rfm_scatter_plot.png")
    kmeans_scatter_plot = Image.open("kmeans_scatter_plot.png")
    hierarchical_scatter_plot = Image.open("hierarchical_scatter_plot.png")

    # Display the images in two rows, three columns
    st.write("##### Comparison of Tree Maps")
    col1, col2, col3 = st.columns(3)

    col1.image(rfm_tree_map, caption="RFM Tree Map", use_column_width=True)
    col2.image(kmeans_tree_map, caption="K-Means Tree Map", use_column_width=True)
    col3.image(hierarchical_tree_map, caption="Hierarchical Tree Map", use_column_width=True)

    st.write("##### Comparison of Scatter Plots")
    col4, col5, col6 = st.columns(3)

    col4.image(rfm_scatter_plot, caption="RFM Scatter Plot", use_column_width=True)
    col5.image(kmeans_scatter_plot, caption="K-Means Scatter Plot", use_column_width=True)
    col6.image(hierarchical_scatter_plot, caption="Hierarchical Scatter Plot", use_column_width=True)

    st.write("##### Summary: ")
    st.markdown("""
    After analyzing the segmentation results, it's evident that both K-Means and Hierarchical Clustering methods yield similar segmentation outcomes. However, K-Means stands out with its faster training time and superior silhouette score.

    In terms of clustering quality, the manual RFM segmentation, based on domain knowledge, proves to be the most effective approach. It divides the dataset into distinct segments, each with a reasonably balanced size.

    In summary, the K-Means clustering model is the preferred choice for making predictions due to its efficient training and competitive performance.
    """)


elif selected_option == 'New Prediction':
    st.subheader("New Prediction")
    
    valid_input = False

    while not valid_input:
        customer_id = st.text_input(label="Enter Customer ID:", key="customer_id_input")
        transaction_date = st.text_input(label="Enter Transaction Date (19970101 to 19980630, YYYYMMDD):", key="transaction_date_input")
        num_cds_purchased = st.number_input(label="Enter Number of CDs Purchased:", min_value=1, step=1, key="num_cds_purchased_input")
        transaction_value = st.number_input(label="Enter Transaction Value (greater than 0.0):", min_value=0.01, format="%.2f", key="transaction_value_input")

        # Validate transaction_date format and range
        # Define the date range
        start_date = pd.Timestamp('1997-01-01')
        end_date = pd.Timestamp('1998-06-30')

        # Validate transaction_date within the specified range
        valid_date = False
        try:
            transaction_date = pd.to_datetime(str(transaction_date), format='%Y%m%d')
            if start_date <= transaction_date <= end_date:
                valid_date = True
        except ValueError:
            st.warning("Please enter a valid transaction date in the format YYYYMMDD and in range 19970101 to 19980630.")
            st.stop()

        # Validate customer_id
        valid_customer_id = False
        if re.match(r'^\d{1,5}$', customer_id):
            valid_customer_id = True
        else:
            st.warning("Customer ID should be a numeric string with a maximum of 5 characters.")


        # Validate num_cds_purchased and transaction_value
        valid_values = True
        if num_cds_purchased < 1:
            st.warning("Number of CDs purchased should be greater than or equal to 1.")
            valid_values = False
        if transaction_value <= 0:
            st.warning("Transaction value should be greater than 0.0.")
            valid_values = False

        if not valid_date or not valid_customer_id or not valid_values:
            st.warning("Please fill in all fields with valid values.")
        else:
            valid_input = True

    # Only display the "Submit" button if input is valid
    if valid_input:
        if st.button("Submit"):
            # Create a DataFrame from the user's input
            user_input_df = pd.DataFrame({
                'customer_id': [customer_id],
                'transaction_date': [transaction_date],
                'transaction_value': [transaction_value],
                'num_cds_purchased': [num_cds_purchased]
            })

            # Process the user input data
            # Convert 'transaction_date' to datetime64 format
            user_input_df['transaction_date'] = pd.to_datetime(user_input_df['transaction_date'], format='%Y%m%d')

            df_RFM = user_input_df.groupby('customer_id').agg({
                'transaction_date': lambda x: (end_date - x.max()).days,
                'transaction_value': ['count', 'sum']
            })

            df_RFM.columns = ['Recency', 'Frequency', 'Monetary']

            # Load the K-Means model
            kmeans_model = joblib.load("kmeans_model.pkl")

            # Predict cluster for the user's data
            user_clusters = kmeans_model.predict(df_RFM)
            
            # Add the predicted cluster to the df_RFM DataFrame
            df_RFM['Cluster'] = user_clusters
            
            st.write("Predicted Cluster:", user_clusters[0])
 

