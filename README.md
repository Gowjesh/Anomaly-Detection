# Log Anomaly Detection Using Machine Learning

A Machine Learning-based Log Anomaly Detection System that identifies unusual or suspicious patterns in system logs.

The project uses TF-IDF to convert log messages into numerical features and applies unsupervised anomaly detection algorithms such as:

* Isolation Forest
* Local Outlier Factor (LOF)
* One-Class SVM

A Streamlit web application is provided for testing individual log messages and performing batch analysis on CSV files.

## Project Overview

System logs contain large amounts of information about the behavior and health of applications and servers. Manually checking these logs for unusual events can be difficult and time-consuming.

This project automatically analyzes log messages and detects patterns that are different from normal logs.

### Workflow

```text
HDFS Log Dataset
       ↓
Data Loading
       ↓
Log Content Extraction
       ↓
TF-IDF Vectorization
       ↓
Feature Scaling
       ↓
Anomaly Detection
       ↓
Normal / Anomaly
       ↓
Streamlit Dashboard
```

## Features

* Detect anomalies in system log messages
* Single log message prediction
* Batch CSV log analysis
* TF-IDF text feature extraction
* Multiple anomaly detection algorithms
* Adjustable contamination rate
* Anomaly score display
* PCA visualization
* Interactive Streamlit interface
* Save trained models using Joblib

## Machine Learning Algorithms

### 1. Isolation Forest

Isolation Forest identifies unusual data points by isolating them from the rest of the data.

```python
IsolationForest(
    contamination=0.1,
    random_state=42
)
```

The model output is converted into:

```text
0 → Normal
1 → Anomaly
```

### 2. Local Outlier Factor

LOF identifies observations that have a significantly different local density compared with their neighbors.

```python
LocalOutlierFactor(
    contamination=0.1
)
```

### 3. One-Class SVM

One-Class SVM learns the boundary of normal data and identifies observations outside that boundary as anomalies.

```python
OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.1
)
```

## Text Processing

The log messages are taken from the `Content` column of the HDFS dataset.

TF-IDF converts the text into numerical features:

```python
vectorize = TfidfVectorizer(max_features=500)

X = vectorize.fit_transform(text)
```

The features are then scaled using StandardScaler:

```python
scaler = StandardScaler(with_mean=False)

X_scaled = scaler.fit_transform(X)
```

## Visualization

PCA is used to reduce the feature space to two dimensions for visualization.

The project generates visualizations for:

* Isolation Forest
* Local Outlier Factor
* One-Class SVM

## Streamlit Application

The application provides two main sections.

### Single Log Check

Users can enter a log message and click Detect Anomaly.

The application returns:

```text
Normal Log
```

or

```text
Anomaly Detected
```

### Batch File Analysis

Users can upload a CSV file containing a `Content` column.

The application displays:

* Total Logs
* Normal Logs
* Anomalies
* Filtered results

## Model Settings

The Streamlit application allows users to select:

```text
Isolation Forest
One-Class SVM
```

The contamination rate can be adjusted between:

```text
0.01 → 0.30
```

## Project Structure

```text
Log-Anomaly-Detection/
│
├── app.py
├── model.py
├── HDFS_2k.log_structured.csv
├── models/
│   ├── isolation_forest.pkl
│   ├── tfidf_vectorizer.pkl
│   └── pca.pkl
│
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* TF-IDF
* Isolation Forest
* Local Outlier Factor
* One-Class SVM
* PCA
* Joblib
* Streamlit

## Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
cd Log-Anomaly-Detection
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn joblib streamlit
```

## Run the Application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## Example Log Inputs

Normal log:

```text
PacketResponder 1 for block blk_38865049064139660 terminating
```

Another log:

```text
BLOCK* NameSystem.addStoredBlock: blockMap updated
```

Potentially suspicious log:

```text
FATAL: unexpected EOF connection lost during block transfer
```

## Applications

This type of system can be useful for:

* Server monitoring
* System log analysis
* Fault detection
* Cybersecurity monitoring
* Distributed system monitoring
* Application monitoring
* IT infrastructure monitoring

## Future Improvements

* Add deep learning-based anomaly detection
* Use LSTM or Autoencoder models
* Add real-time log monitoring
* Store detected anomalies in a database
* Add email alerts
* Improve model evaluation using labeled datasets
* Deploy the application online
* Add authentication and user management

## OUTPUT

<img width="1897" height="906" alt="image" src="https://github.com/user-attachments/assets/00ad5834-1af6-4a41-ada4-0d79e38246e1" />
