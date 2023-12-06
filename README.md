# Real-time Sensor Data Analysis and Emotional State Prediction Dashboard

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
Certainly! Here's a "How to Run" section for your GitHub repository:

---

## How to Run

### 1. Download WESAD Dataset
Download the WESAD dataset from the provided link: [WESAD Dataset](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)

Ensure the dataset is saved and unzipped with the structure: `WESAD\S2\...`

### 2. Install Dependencies
Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

### 3. Install Grafana Dashboard
Install Grafana Dashboard using the appropriate instructions for your operating system. You can find installation guides on the official Grafana website: [Grafana Installation Guide](https://grafana.com/docs/grafana/latest/installation/)

### 4. Data Preparation
Run the following scripts for data preparation:

- **Run `DataPrep.py`**:
  ```bash
  python dataprep.py
  ```
  This will create the `MergedData` required for further analysis.

- **Run `parse_readme.py`**:
  ```bash
  python parse_readme.py
  ```
  This script completes information in the merged data.

### 5. Exploratory Data Analysis (EDA)
Perform exploratory data analysis using the following scripts:

- **Run `spark_eda.py`**:
  ```bash
  python spark_eda.py
  ```
  This script generates correlation matrices and other exploratory analyses using PySpark.

- **Run `Pandas_EDA.ipynb`**:
  ```bash
  python pandas_eda.py
  ```
  This script performs exploratory data analysis using Pandas.

### 6. Model Training and Tuning
Train and tune machine learning models:

- **Run `spark_model_tuning.py`**:
  ```bash
  python spark_model_tuning.py
  ```
  This script performs hyperparameter tuning for the Random Forest algorithm using PySpark.

- **Run `ml_training.py`**:
  ```bash
  python ml_training.py
  ```
  This script trains the TensorFlow model and generates logs in the `Logs/logs` directory.

### 7. Prediction
Generate predictions over a randomly selected subset of data:

- **Run `predict.py`**:
  ```bash
  python predict.py
  ```
  This script provides predictions based on the trained models.

---

Ensure you follow the steps in order to successfully run the project. If you encounter any issues, refer to the documentation or reach out to the project contributors for assistance.
