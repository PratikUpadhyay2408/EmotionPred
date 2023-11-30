import os
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pathlib import Path


def load_data( spark ,data_dir, subject_list=None, load_individual_df=False, sample_factor=None):

    # Initialize an empty PySpark DataFrame
    full_df = None

    # load only a single df
    if load_individual_df:
        data_source = [ os.path.join( data_dir, f"{subject}.parquet") for subject in subject_list ]
    else:
        data_source = Path(data_dir).glob('*.parquet')

    # Iterate through Parquet files and read them into the PySpark DataFrame
    for parquet_file in data_source:

        if not os.path.isfile(parquet_file):
            print( f"Subject data {parquet_file} not found" )
            return None

        print(f"reading {parquet_file}")
        # Read the current Parquet file into a PySpark DataFrame
        current_df = spark.read.parquet(str(parquet_file))
        # Union the current DataFrame with the existing one
        if full_df is None:
            full_df = current_df
        else:
            full_df = full_df.union(current_df)

    if sample_factor:
        sampled_df = full_df.sampleBy('sid', fractions=sample_factor, seed =42 )
    else:
        sampled_df = full_df

    # Show the resulting PySpark DataFrame
    sampled_df.show(n=5)
    return sampled_df


def transform_data( df ):
    feature_columns = ["c_acc_x", "c_acc_y", "c_acc_z", "ecg", "emg", "c_eda", "c_temp",
                       "resp", "w_acc_x", "w_acc_y", "w_acc_z", "w_eda", "bvp", "w_temp"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    vectorized_df = assembler.transform(df)
    return vectorized_df


def calc_corr(spark_df, image):

    print( "calculate correlation matrix for spark dataframe" )
    columns_to_drop = ["sid", "label"]
    corr_df = spark_df.drop(*columns_to_drop)
    # Convert the remaining columns to a single vector column
    vectorized_df = transform_data(spark_df)
    # Compute the correlation matrix
    correlation_matrix = Correlation.corr(vectorized_df, "features").head()
    # Extract the correlation matrix as a NumPy array
    corr_matrix_np = correlation_matrix[0].toArray()
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    # Create a heatmap using seaborn
    sns.heatmap(corr_matrix_np, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                xticklabels=corr_df.columns, yticklabels=corr_df.columns)
    # Add a title
    plt.title('Correlation Matrix Heatmap')

    # Show the plot
    print( f"saving correlation matrix as {image}" )
    plt.savefig( os.path.join("Results",image) )


def rf_classifier( spark_df, label_col ):

    # Assemble features into a single vector column
    vectorized_df = transform_data(spark_df)

    # Split the data into training and testing sets
    train_ratio = 0.8
    test_ratio = 1.0 - train_ratio
    train_data, test_data = vectorized_df.randomSplit([train_ratio, test_ratio], seed=42)

    # Define the rf classifier
    rf = RandomForestClassifier(labelCol=label_col, featuresCol="features", maxDepth=7)

    # Train the rf model
    model = rf.fit(train_data)

    # Make predictions on the test set
    predictions = model.transform(train_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    print(f"train F1 Score: {f1_score}")
    print(f"train accuracy: {model.summary.accuracy}")  # summary only

    # Make predictions on the test set
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    print(f"test F1 Score: {f1_score}")
    print(f"test accuracy: {model.summary.accuracy}")  # summary only

    return model, train_data, test_data


def get_feature_importance(model, image):
    # Get feature importance scores
    feature_col = ["c_acc_x", "c_acc_y", "c_acc_z", "ecg", "emg", "c_eda", "c_temp",
                       "resp", "w_acc_x", "w_acc_y", "w_acc_z", "w_eda", "bvp", "w_temp"]

    feature_importance = model.featureImportances

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_col)), feature_importance, align='center')
    plt.xticks(range(len(feature_col)), feature_col, rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance')

    plt.savefig( os.path.join("Results",image) )


def load_fractional_data(spark, data_dir, fraction=0.001):
    # call function to load data for all subjects with a sample of 5% from each subject
    sample_dict = {}
    for i in range(2, 18):
        if i != 12:
            sample_dict[i] = fraction
    all_df = load_data(spark, data_dir, sample_factor=sample_dict)
    all_df = all_df.repartition(100)
    return all_df


if __name__ == "__main__":

    # Create a Spark session
    spark = SparkSession.builder.appName("parquet_reader").getOrCreate()
    # Set the path to the directory containing the Parquet files
    data_dir = "MergedData"
    all_df = load_fractional_data(spark, data_dir)
    #plot correlation matrix
    calc_corr(all_df, r"Correlation_Matrix.png")
    # calculate feature importance
    model, train_data, test_data = rf_classifier(all_df, "label")
    # plot feature importance
    get_feature_importance(model, r"Feature_Importance")


    spark.stop()

