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
import tempfile

pio.templates.default = 'plotly' 

transactions = []
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

    
def rfm_level2(df, num_clusters):
    if num_clusters == 2:
        if (df['RFM_score'] >= 9):
            return 'Active Customers'
        else:
            return 'Lost Customers'
    
    if num_clusters == 3:
        if (df['RFM_score'] == 12):
            return 'VIP Customers'
    
        elif (df['R'] >= 3):
            return 'Active Customers'
        else:
            return 'Lost Customers'
    
    if num_clusters == 4:
        if (df['RFM_score'] == 12):
            return 'VIP Customers'
        elif (df['R'] >= 2 and df['F'] == 4):
            return 'Potential Loyalists'
        elif (df['F'] <= 2 and df['M'] <= 2 and df['R'] <= 2):
            return 'Lost'
        else:
            return 'Regular'
        
    elif num_clusters == 5:
        # Define conditions for 5 clusters
        # Add more conditions as needed
        if (df['RFM_score'] == 12):
            return 'Group 1'
        elif (df['R'] == 4):
            return 'Group 2'
        elif (df['M'] == 4):
            return 'Group 3'
        elif (df['F'] == 4):
            return 'Group 4'
        else:
            return 'Group 5'
    elif num_clusters == 6:
        # Define conditions for 6 clusters
        if (df['RFM_score'] == 12):
            return 'Group 1'
        elif (df['R'] == 1 and (df['F'] == 1 and df['M'] == 1)):
            return 'Group 2'
        elif (df['M'] == 4):
            return 'Group 3'
        elif (df['R'] >= 3 and df['M'] >= 3):
            return 'Group 4'
        elif (df['F'] >= 3 and df['R'] >= 3):
            return 'Group 5'
        else:
            return 'Group 6'
    elif num_clusters == 7:
        if (df['RFM_score'] == 12):
            return 'Group 1'
        elif (df['R'] == 4):
              return 'Group 2'
        elif (df['F'] <= 2 and df['M'] <= 2):
            return 'Group 3'
        elif (df['M'] == 4):
            return 'Group 4'
        elif (df['R'] >= 3 and df['F'] <= 2):
            return 'Group 5'
        elif (df['F'] >= 3 and df['R'] >= 2):
            return 'Group 6'
        else:
            return 'Group 7'

    
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
    df_RFM['RFM_level'] = df_RFM.apply(lambda row: rfm_level2(row, num_clusters), axis=1)
    # if num_clusters == 2:
    #     df_RFM['RFM_level'] = df_RFM.apply(rfm_level, axis=1)
    # else:
    #     df_RFM['RFM_level'] = df_RFM.apply(rfm_level1, axis=1)

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
    
    fig2.savefig(f'rfm_tree_map_{num_clusters}_clusters.png',bbox_inches='tight')  # Save the figure
    st.write("##### RFM Customer Segments Tree Map")
    st.image(f"rfm_tree_map_{num_clusters}_clusters.png")  # Display the saved image

     # Create a color scale (e.g., viridis, rainbow, jet, etc.)
    color_scale = px.colors.sequential.Viridis  # You can change this to your preferred color scale

    fig3 = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_level",
                      hover_name="RFM_level", size_max=100, color_continuous_scale=color_scale)
    
    # Save the scatter plot as an image file using Plotly's to_image method
    scatter_image = pio.to_image(fig3, format="png", width=800, height=600, scale=1)

    #fig3.write_image("RFM_Scatter.png")  # Save the scatter plot as an image
    st.write("##### RFM Customer Segments Scatter Plot")
    # Save the image to a file
    scatter_image_file = f"rfm_scatter_plot_{num_clusters}_clusters.png"
    with open(scatter_image_file, "wb") as img_file:
        img_file.write(scatter_image)

    st.image(scatter_image_file)
    
    # Add a new column "Number of Clusters" to result_data
    rfm_results["Number of Clusters"] = num_clusters

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
    joblib.dump(kmeans, f'kmeans_model_{k}_clusters.pkl')

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

    # # Build the K-Means model with the specified number of clusters
    # kmeans = build_kmeans_model(rfm_scaled, num_clusters)
    # # Create a new DataFrame with the original data and cluster assignments
    # clustered_data = pd.DataFrame({'customer_id': df_RFM.index, 'Cluster': kmeans.labels_})

    # Build the K-Means model with the specified number of clusters
    kmeans = joblib.load(f"kmeans_model_{num_clusters}_clusters.pkl")
    clustered_data  = kmeans.predict(rfm_scaled)

    # Convert clustered_data NumPy array to a DataFrame
    clustered_data_df = pd.DataFrame({'customer_id': df_RFM.index, 'Cluster': kmeans.labels_})
    
    # End timing and calculate the time taken
    end_time = time.time()
    time_taken = end_time - start_time


    # Merge with the original data
    result_data = pd.merge(rfm_selected, clustered_data_df, left_index=True, right_on='customer_id')
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
    fig1.savefig(f"kmeans_tree_map_{num_clusters}_clusters.png", bbox_inches='tight')

    # Display the saved tree map using st.image
    st.image(f"kmeans_tree_map_{num_clusters}_clusters.png")

    # Create a color scale (e.g., viridis, rainbow, jet, etc.)
    color_scale = px.colors.sequential.Viridis  # You can change this to your preferred color scale

    # Scatter plot with the chosen color scale
    fig2 = px.scatter(rfm_agg2, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
                    hover_name="Cluster", size_max=100, color_continuous_scale=color_scale)

    # Save the scatter plot as an image file using Plotly's to_image method
    scatter_image = pio.to_image(fig2, format="png", width=800, height=600, scale=1)

    # Save the image to a file
    with open(f"kmeans_scatter_plot_{num_clusters}_clusters.png", "wb") as img_file:
        img_file.write(scatter_image)

    st.write("##### KMeans Customer Segments Scatter Plot")
    # Display the saved scatter plot using st.image
    st.image(f"kmeans_scatter_plot_{num_clusters}_clusters.png")
    
    # Add a new column "Number of Clusters" to result_data
    rfm_agg2["Number of Clusters"] = num_clusters

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
    fig1.savefig(f"py_kmeans_tree_map_{num_clusters}_clusters.png", bbox_inches='tight')

    # Display the saved image using st.image
    st.image(f"py_kmeans_tree_map_{num_clusters}_clusters.png")

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
    scatter_image_file = f"hierarchical_scatter_plot_{num_clusters}_clusters.png"
    with open(scatter_image_file, "wb") as img_file:
        img_file.write(scatter_image)

    st.image(scatter_image_file)

    # Save the KMeans model
    kmeans_model.save(f"pyspark_kmeans_model_{num_clusters}_cluster.pkl")

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
    # agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    
    
    # Load the Hierachical model with the specified number of clusters
    agg_clustering = joblib.load(f"hierarchical_clustering_model_{num_clusters}_clusters.pkl")
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
    squarify_fig.savefig(f"hierarchical_tree_map_{num_clusters}_clusters.png", bbox_inches='tight')
    st.write("###### Hierachical Clustering Tree Map")
    st.image(f"hierarchical_tree_map_{num_clusters}_clusters.png")

    # Create a color scale (e.g., viridis, rainbow, jet, etc.)
    color_scale = px.colors.sequential.Viridis  # You can change this to your preferred color scale

    # Scatter plot with the chosen color scale
    scatter_fig = px.scatter(rfm_agg3, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
                            hover_name="Cluster", size_max=100, color_continuous_scale=color_scale)

    st.write("###### Hierachical Clustering Scatter Plot")

    # Save the scatter plot as an image file using Plotly's to_image method
    scatter_image = pio.to_image(scatter_fig, format="png", width=800, height=600, scale=1)

    # Save the image to a file
    scatter_image_file = f"hierarchical_scatter_plot_{num_clusters}_clusters.png"
    with open(scatter_image_file, "wb") as img_file:
        img_file.write(scatter_image)

    st.image(scatter_image_file)

    # Save the Hierarchical Clustering model to a file
    with open(f"hierarchical_clustering_model_{num_clusters}_clusters.pkl", "wb") as model_file:
        pickle.dump(agg_clustering, model_file)

    # Add a new column "Number of Clusters" to result_data
    rfm_agg3["Number of Clusters"] = num_clusters

    # Return results
    return rfm_agg3, execution_time, silhouette

# Define a function to update or add records in global_results_df

def update_global_results(num_clusters, model_name, execution_time, silhouette_score, csv_filename='comparison.csv'):
    # Load the existing comparison data from the CSV file
    try:
        global_results_df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty DataFrame
        global_results_df = pd.DataFrame(columns=["Number of Clusters", "Model", "Execution Time (s)", "Silhouette Score"])

    # # Check if the model name already exists in the DataFrame
    # existing_index = global_results_df[global_results_df["Model"] == model_name].index

    # Check if the combination of model name and "Number of Clusters" exists
    existing_index = global_results_df[
        (global_results_df["Model"] == model_name) & (global_results_df["Number of Clusters"] == num_clusters)
    ].index

    if not existing_index.empty:
        # Replace the existing record with the same model name
        global_results_df.loc[existing_index, ["Execution Time (s)", "Silhouette Score"]] = [execution_time, silhouette_score]
    else:
        # Add a new record for the model
        new_record = {"Number of Clusters": num_clusters,"Model": model_name, "Execution Time (s)": execution_time, "Silhouette Score": silhouette_score}
        global_results_df = pd.concat([global_results_df, pd.DataFrame([new_record])], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    global_results_df.to_csv(csv_filename, index=False)

# Create a global DataFrame to store results
global_results_df = pd.DataFrame(columns=["Model", "Execution Time (s)", "Silhouette Score"])

data_path = 'CDNOW_master/CDNOW_master.txt'
df = load_data(data_path)
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
    
    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=['txt'])

    if uploaded_file is not None:
        # Create empty lists to store column data
        customer_ids = []
        transaction_dates = []
        num_cds_purchased = []
        transaction_values = []

        try:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Reset the file pointer to the beginning
            uploaded_file.seek(0)

            # Read the temporary file line by line and split values by whitespace
            with open(temp_file_path, 'r') as file:
                for line in file:
                    #line = line.strip().replace('"', '') 
                    values = line.strip().split()
                    if len(values) == 4:  # Check for 4 columns in your data
                        # Assign values to respective columns
                        #st.write(len(values))
                        customer_ids.append(values[0])
                        transaction_dates.append(pd.to_datetime(values[1], format='%Y%m%d'))  # Assuming date is in the third column
                        num_cds_purchased.append(pd.to_numeric(values[2]))  # Assuming quantity is in the fourth column
                        transaction_values.append(pd.to_numeric(values[3]))  # Assuming value is in the fifth column

            # Create a DataFrame from the lists
            df = pd.DataFrame({
                'customer_id': customer_ids,
                'transaction_date': transaction_dates,
                'num_cds_purchased': num_cds_purchased,
                'transaction_value': transaction_values
            })

            # Display the shape of the loaded DataFrame
            st.write("Loaded DataFrame Shape:", df.shape)

            # Display a sample of the loaded data
            st.subheader("Sample of Loaded Data:")
            st.write(df.head())

            # Save the DataFrame to a new tab-separated text file
            df.to_csv("CDNOW_master_new.txt", sep='\t', index=False)

        except FileNotFoundError:
            st.warning("Uploaded file not found.")

        finally:
            # Delete the temporary file after reading
            os.remove(temp_file_path)
    else:
        st.write("There is no file uploaded")

    #------------------
    # # Create a dropdown or radio button widget to select the number of clusters
    # num_clusters = st.radio("Select Number of Clusters", [2, 3])
    # Create a text input for the number of clusters
    num_clusters = st.text_input("Enter Number of Clusters", "3")  # Default value is "3"

    # Convert the input to an integer
    num_clusters = int(num_clusters)  # Assuming the input is a valid integer

    # Check if the input is within the range of 2 to 7
    if 2 <= num_clusters <= 7:
        st.success(f"Number of Clusters: {num_clusters}")
    else:
        st.error("Please enter a value between 2 and 7 for the number of clusters.")


    st.subheader("Build Project")

    st.write("### 1. Data Exploration and Prepraration")
    explore_data(df)
    df_RFM, current_date = prepare_data(df)

    st.write("### 2. Build Models")
    
    # Perform RFM analysis
    if st.button("Perform RFM Analysis",key="rfm"):
        rfm_results, rfm_time = perform_rfm_analysis(df_RFM,num_clusters)
        
        # Call the function to update or add the record
        update_global_results(num_clusters,"RFM Analysis", rfm_time, 0)

        # Define the file path where you want to save the CSV file
        csv_file_path = "rfm_results.csv"
       
        # Use the to_csv method to write the DataFrame to a CSV file
        rfm_results.to_csv(csv_file_path, index=False)

    # Perform K-Means clustering
    if st.button("Perform K-Means Clustering",key="kmeans"):
        rfm_selected = df_RFM[['Recency','Frequency','Monetary']]
        kmeans_results, kmean_time, kmean_silhouette  = perform_kmeans_clustering(rfm_selected,num_clusters)
        
        update_global_results(num_clusters,"KMean Clustering", kmean_time, kmean_silhouette)

        # Define the file path where you want to save the CSV file
        csv_file_path = "kmeans_results.csv"

        # Use the to_csv method to write the DataFrame to a CSV file
        kmeans_results.to_csv(csv_file_path, index=False)

        
    # # Perform Hierachical Clustering
    if st.button("Perform Hierachical Clustering",key="tree"):
        rfm_selected = df_RFM[['Recency','Frequency','Monetary']]
        tree_results, tree_time, tree_silhouette = perform_hierarchical_clustering(rfm_selected,num_clusters)
        update_global_results(num_clusters,"Hierachical Clustering", tree_time, tree_silhouette)
        # Define the file path where you want to save the CSV file
        csv_file_path = "tree_results.csv"

        # Use the to_csv method to write the DataFrame to a CSV file
        tree_results.to_csv(csv_file_path, index=False)

    # # Perform K-Means clustering pyspark
    # if st.button("Perform Pyspark K-Means Clustering"):
    #     rfm_selected = df_RFM[['Recency','Frequency','Monetary']]
    #     py_kmeans_results, py_kmean_time, py_kmean_silhouette  = perform_pyspark_kmeans_clustering(rfm_selected)
    #     update_global_results(num_clusters,"Pyspark KMeans Clustering", py_kmean_time, py_kmean_silhouette)

elif selected_option == 'Model Selection':
    # Add a slider to select the number of clusters (from 2 to 7)
    num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=7, value=4)

    # Add a "Select Model" button
    if st.button("Select Model"):  
        # Check if the "comparison.csv" file exists
        if os.path.isfile('comparison.csv'):
            # Load the comparison data from the CSV file
            comparison_df = pd.read_csv('comparison.csv')
            comparison_df = comparison_df[comparison_df["Number of Clusters"] == num_clusters]
            comparison_df = comparison_df.drop("Number of Clusters", axis=1)
            
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
        combined_results = combined_results[combined_results["Number of Clusters"] == num_clusters]
        combined_results = combined_results.drop("Number of Clusters", axis=1)

        if not combined_results.empty:
            # Display the comparison data
            st.write("##### Comparision of clustering results")
            st.dataframe(combined_results)
        else:
            st.warning("No records found in the clustering results data. Please select the 'Build Project' menu to generate results.")
    
        # Load the tree map images
        rfm_tree_map = Image.open(f"rfm_tree_map_{num_clusters}_clusters.png")
        kmeans_tree_map = Image.open(f"kmeans_tree_map_{num_clusters}_clusters.png")
        hierarchical_tree_map = Image.open(f"hierarchical_tree_map_{num_clusters}_clusters.png")

        # Load the scatter plot images
        rfm_scatter_plot = Image.open(f"rfm_scatter_plot_{num_clusters}_clusters.png")
        kmeans_scatter_plot = Image.open(f"kmeans_scatter_plot_{num_clusters}_clusters.png")
        hierarchical_scatter_plot = Image.open(f"hierarchical_scatter_plot_{num_clusters}_clusters.png")

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

elif selected_option == "New Prediction":
    st.subheader("New Prediction")

    # Add a slider to select the number of clusters (from 2 to 7)
    num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=7, value=4)

    # Initialize the session state to store transaction data
    if 'transactions' not in st.session_state:
        st.session_state.transactions = pd.DataFrame(columns=['transaction_date', 'num_cds_purchased', 'transaction_value'])

    # Create a radio button for selecting data input option
    data_input_option = st.radio("Select Data Input Option", ("Input Data", "Load Data"), index=0)  # Default to "Input Data"
    start_date = pd.Timestamp(datetime(1997, 1, 1))
    end_date = pd.Timestamp(datetime(1998, 6, 30))
    # Check the selected option
    if data_input_option == "Input Data":
        with st.form('Input_Data'):
            # User input fields
            transaction_date = st.date_input("Transaction Date", end_date)
            cds_purchased = st.number_input("Number of CDs Purchased", min_value=1, step=1, format="%d")
            transaction_value = st.number_input("Transaction Value ($)", min_value=10.0, step=0.50)

            # Perform validation
            transaction_date = pd.Timestamp(transaction_date)  # Convert to Pandas Timestamp
            valid_date = start_date <= transaction_date <= end_date
            # Check if the transaction date is within the specified range
            valid_cds = isinstance(cds_purchased, int) and cds_purchased > 0
            valid_amount = isinstance(transaction_value, float) and transaction_value > 0.0

            # Check if the form is submitted
            if st.form_submit_button('Submit'):
                if not valid_date:
                    st.warning(f"Transaction date must be between {start_date.date()} and {end_date.date()}.")
                elif not valid_cds:
                    st.warning("Number of CDs should be greater than 0")
                elif not valid_amount:
                    st.warning("Transaction value should be greater than 0")
                else:
                    if valid_date and valid_cds and valid_amount:
                        # Create a new DataFrame with the user's input
                        new_transaction = pd.DataFrame({
                            'transaction_date': [transaction_date],
                            'num_cds_purchased': [cds_purchased],
                            'transaction_value': [transaction_value]
                        })

                        # Append the new transaction data to the session state transactions DataFrame
                        st.session_state.transactions = pd.concat([st.session_state.transactions, new_transaction], ignore_index=True)
                        st.success("Transaction added successfully!")
        
        # Create a checkbox for the reset action
        reset_data = st.toggle("Reset Transactions")
        # Check if the reset checkbox is selected
        if reset_data:
            st.session_state.transactions = pd.DataFrame(columns=['transaction_date', 'num_cds_purchased', 'transaction_value'])
            st.success("Transactions cleared!")

        if not st.session_state.transactions.empty: 
            # Display the current transactions DataFrame
            st.subheader("Current Transactions")
            st.write(st.session_state.transactions)
            # # Assuming that 'Transaction Date' is a column in the DataFrame st.session_state.transactions
            # max_transaction_date = st.session_state.transactions['transaction_date'].max()
            # recency = (end_date - max_transaction_date).days
            # st.write("Recency: ",recency)

            # # Display the maximum transaction date
            # st.write(f"Max Transaction Date: {max_transaction_date}")


        # Create a button to Customer Segment
        if st.button("Customer Segment"):
            if not st.session_state.transactions.empty:
                user_input_df = st.session_state.transactions
                user_input_df['customer_id'] = '00001'
                df_RFM = user_input_df.groupby('customer_id').agg({
                    'transaction_date': lambda x: (end_date - x.max()).days,
                    'transaction_value': ['count', 'sum']
                })

                df_RFM.columns = ['Recency', 'Frequency', 'Monetary']

                st.dataframe(df_RFM)

                # Scale the features
                scaler = StandardScaler()
                rfm_scaled = scaler.fit_transform(df_RFM)

                kmeans_model = joblib.load(f"kmeans_model_{num_clusters}_clusters.pkl")
                user_clusters = kmeans_model.predict(df_RFM[['Recency', 'Frequency', 'Monetary']])
                # Format the message as a single string
                success_message = f"Predicted Clusters: {user_clusters}"

                # Display the success message
                st.success(success_message)
            else:
                st.warning("No transactions to process.")
    else:
        # Add code to load data from a file when "Load Data" is selected
        # You can use st.file_uploader or any other method to load data
        # Upload file
        uploaded_file = st.file_uploader("Choose a file", type=['txt','csv'])
        #st.write(uploaded_file.name)
        if uploaded_file is not None:
            file_name = uploaded_file.name
            # Load and preprocess the data
            df = load_data(file_name)
            st.success(f"{len(df)} transactions are loaded successfully")
            # Convert 'transaction_date' to datetime64 format
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%Y%m%d')
    
        if st.button("Customer Segment"):
            df_RFM = df.groupby('customer_id').agg({
                    'transaction_date': lambda x: (end_date - x.max()).days,
                    'transaction_value': ['count', 'sum']
                })

            df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
            # Select the columns to use for clustering
            rfm_selected = df_RFM[['Recency', 'Frequency', 'Monetary']]

            # Scale the features
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_selected)

            
            # Build the K-Means model with the specified number of clusters
            kmeans_model = joblib.load(f"kmeans_model_{num_clusters}_clusters.pkl")
            new_cluster_labels  = kmeans_model.predict(rfm_scaled)

            # Add cluster labels to the DataFrame
            df_RFM['Cluster'] = new_cluster_labels

            # Display segmentation results
            st.write("##### KMeans Customer Segmentation Sample Results")
            st.dataframe(df_RFM)

            # Write result clustering to csv
            df_RFM.to_csv('prediction_result_data.csv', index=False)


            # Calculate average values for each cluster
            rfm_agg2 = df_RFM.groupby('Cluster').agg({
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

            #Visualization
            # Get cluster centroids
            # centroids = kmeans_model.cluster_centers_

            # # Create a scatter plot
            # fig, ax = plt.subplots(figsize=(8, 6))

            # # Plot data points
            # scatter = ax.scatter(rfm_scaled['Recency'], rfm_scaled['Frequency'], c=rfm_scaled['Cluster'], cmap='viridis', label='Data Points')

            # # Plot cluster centroids
            # ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Cluster Centroids')

            # ax.set_title('Cluster Visualization with Centroids')
            # ax.set_xlabel('Recency')
            # ax.set_ylabel('Frequency')
            # ax.legend()

            # # Display the plot in Streamlit
            # st.pyplot(fig)

            # Create a separate figure for the tree map
            fig1, ax_1 = plt.subplots(figsize=(14, 10))

            colors_dict2 = {'Cluster0': 'yellow', 'Cluster1': 'royalblue', 'Cluster2': 'cyan',
                            'Cluster3': 'red', 'Cluster4': 'purple', 'Cluster5': 'green', 'Cluster6': 'gold'}

            squarify.plot(sizes=rfm_agg2['Count'],
                        text_kwargs={'fontsize': 12, 'weight': 'bold', 'fontname': "sans serif"},
                        color=colors_dict2.values(),
                        label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2.iloc[i])
                                for i in range(0, len(rfm_agg2))], alpha=0.5)

            plt.title("Customers Segments Tree Map", fontsize=26, fontweight="bold")
            plt.axis('off')

            # Save the tree map as an image file
            fig1.savefig("result_tree_map.png", bbox_inches='tight')

            # Display the saved tree map using st.image
            st.image("result_tree_map.png")

            # Create a color scale (e.g., viridis, rainbow, jet, etc.)
            color_scale = px.colors.sequential.Viridis  # You can change this to your preferred color scale

            # Scatter plot with the chosen color scale
            fig2 = px.scatter(rfm_agg2, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
                            hover_name="Cluster", size_max=100, color_continuous_scale=color_scale)

            # Save the scatter plot as an image file using Plotly's to_image method
            scatter_image = pio.to_image(fig2, format="png", width=800, height=600, scale=1)

            # Save the image to a file
            with open("results_scatter_plot.png", "wb") as img_file:
                img_file.write(scatter_image)

            st.write("##### Customer Segments Scatter Plot")
            # Display the saved scatter plot using st.image
            st.image("results_scatter_plot.png")
            





            
