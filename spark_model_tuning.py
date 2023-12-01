import spark_eda as data_proc

from pyspark.sql import SparkSession

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def rf_hyper_param_tuning( train, test ):
    # Create a parameter grid for the RandomForest model

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42)

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [i for i in range(10,30,5)]) \
        .addGrid(rf.maxDepth, [i for i in range(5,15,2)]) \
        .build()


    # Create a multiclass classification evaluator with F1 score
    evaluator = MulticlassClassificationEvaluator(metricName="f1")

    # Create a CrossValidator with the RandomForest model, parameter grid, and evaluator
    crossval = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=5)  # You can adjust the number of folds as needed

    # Run cross-validation to tune the model
    cvModel = crossval.fit(train)

    # Get the best model from cross-validation
    bestModel = cvModel.bestModel

    # Make predictions on the test set
    predictions = bestModel.transform(test)
    # Evaluate the model on the test set using F1 score
    eval_result = evaluator.evaluate(predictions)
    print("F1 score on test data = {}".format(eval_result))

    # Print the best parameters
    bestParams = bestModel.extractParamMap()
    print("Best Parameters:")
    for param, value in bestParams.items():
        print("{}: {}".format(param.name, value))

    mPath = "Results/Tuned_Rf"
    cvModel.bestModel.write().overwrite().save(mPath)


if __name__ == "__main__":

    spark = SparkSession.builder.appName("hyperparam_tuning").getOrCreate()

    data_dir = "MergedData"

    all_df = data_proc.load_fractional_data(spark, data_dir)
    # Assemble features into a single vector column
    vectorized_df = data_proc.transform_data(all_df)

    # Split the data into training and testing sets
    train_ratio = 0.8
    test_ratio = 1.0 - train_ratio
    train_data, test_data = vectorized_df.randomSplit([train_ratio, test_ratio], seed=42)

    rf_hyper_param_tuning( train_data, test_data )

    spark.stop()
