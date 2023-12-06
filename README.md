# Real-time Sensor Data Analysis and Prediction Dashboard**

## Objective
The primary goal of this project is to leverage the WESAD CSV dataset to create a comprehensive real-time sensor data analysis and prediction system. This involves a combination of data cleaning, merging, resampling, and synchronization, regression modeling using PySpark and Python, and integration into a Grafana dashboard for enhanced user awareness.

## Key Tasks

### Data Cleaning, Merging, Resampling, and Synchronization
- Utilize Pandas and SciPy for efficient data cleaning and merging.
- Transform the WESAD CSV dataset into Parquet format.
- Implement resampling techniques to handle varying time intervals.
- Synchronize sensor signals to ensure temporal alignment.

### Exploratory Data Analysis (EDA)
- Conduct initial EDA using Pandas for a correlation matrix.
- Transition to PySpark for scalable EDA.

### Random Forest Classifier and Feature Importance
- Train a Random Forest classifier using PySpark.
- Extract feature importance for valuable insights.

### Neural Network Training with TensorFlow
- Implement a neural network using TensorFlow.
- Monitor Training using Tensorboard
- Save the trained model for future use.

### Prediction System
- Leverage trained models for real-time predictions.
- Analyze incoming data for prompt predictions.

### Grafana Dashboard Development
- Develop a Grafana dashboard for real-time visualization.
- Include live updates of predictions, subject information, and feature plots.
- Generate a PostgreSQL database from a subset of data for visualization.
- Generate csv files to enable fast live updates on the dashboard for subject information and emotional state prediction

## Expected Outcomes

- Efficient data cleaning, transformation into Parquet format, resampling, and synchronization using Pandas and SciPy.
- Thorough exploratory data analysis with Pandas and scalable analysis with PySpark.
- Trained Random Forest classifier and extracted feature importance.
- Implementation of a neural network for enhanced predictive capabilities.
- Real-time predictions using the trained models.
- Development of a Grafana dashboard with live updates and feature plots.

## Benefits

- Improved decision-making with real-time insights and predictive analytics.
- Proactive issue identification through immediate user alerts on the Grafana dashboard.
- Enhanced user experience with an informative dashboard displaying live updates, important features, and synchronized sensor signals.
