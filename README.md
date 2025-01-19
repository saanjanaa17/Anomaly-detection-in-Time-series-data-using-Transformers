# Transformer-Based Autoencoder for Anomaly Detection in Industrial Time-Series Data

## Overview

This project focuses on developing an **unsupervised anomaly detection system** for multivariate time-series data using a **Transformer-based Autoencoder**. The model is designed to detect anomalies in sensor data collected from industrial equipment, particularly from a hydraulic test rig. The goal is to identify deviations from normal patterns without the need for labeled data, making it well-suited for real-world applications in industries like oil & gas, where labeling data can be expensive and impractical.

## Key Features

- **Unsupervised Anomaly Detection**: Detect anomalies in sensor data without the need for labeled data.
- **Transformer-based Autoencoder**: Utilizes the power of Transformer models and Autoencoders for capturing complex patterns and reconstructing time-series data.
- **Positional Encoding**: Temporal dependencies in data are preserved using positional encoding.
- **Efficient Anomaly Identification**: Anomalies are flagged based on high reconstruction error, signaling deviations from normal operational patterns.
- **Flexible Dataset Compatibility**: Supports high-dimensional sensor data, such as temperature, pressure, flow rate, and vibration data from industrial environments.

## Project Structure

The project consists of the following key components:

1. **Data Preprocessing**:
   - Aggregating raw sensor data into 30-minute intervals for analysis.
   - Handling missing data using interpolation to maintain continuous data flow.

2. **Model Architecture**:
   - **Encoder**: Uses 1D convolutional layers, multi-head attention, and global average pooling to capture both local and global patterns in the time series data.
   - **Decoder**: Reconstructs the input data using LSTM layers and time-distributed dense layers.
   - **Positional Encoding**: Preserves temporal information in the input data, ensuring the model understands the sequence of events.

3. **Training & Optimization**:
   - **Unsupervised Learning**: The model is trained on normal patterns without requiring labeled anomalies.
   - **Batch Processing**: Efficient batch processing ensures that large datasets are handled effectively.
   - **Loss Function**: Uses Mean Squared Error (MSE) to minimize reconstruction error and identify anomalies.

4. **Anomaly Detection**:
   - Anomalies are identified when the reconstruction error exceeds a predefined threshold.

5. **Evaluation**:
   - The model’s performance is evaluated based on its ability to reconstruct data accurately, with higher errors signaling anomalies.

## Data

- **Data Source**: Industrial sensor data, including pressure, temperature, flow rate, and vibration sensors, from a hydraulic test rig.
- **Attributes**:
  - **Pressure Sensors (PS1 - PS6)**: 100 Hz
  - **Flow Sensors (FS1, FS2)**: 10 Hz
  - **Temperature Sensors (TS1 - TS4)**: 1 Hz
  - **Vibration Sensor (VS1)**: 1 Hz
  - **Motor Power (EPS1)**: 100 Hz
  - **Cooling Efficiency (CE)**: 1 Hz
  - **Cooling Power (CP)**: 1 Hz
  - **Efficiency Factor (SE)**: 1 Hz
- **Data Format**: Time-series data is structured as matrices with 2205 instances and 43,680 attributes, including various sampling rates (1 Hz, 10 Hz, and 100 Hz).
- **Target Values**: Separate file annotations ('profile.txt') include cooler, valve, pump, and accumulator conditions, as well as stable flags.

## Requirements

To run the project, you'll need the following dependencies:

- **TensorFlow** and **Keras** for model development
- **NumPy**, **Pandas** for data manipulation
- **Matplotlib** and **Seaborn** for visualizations
- **Scikit-learn** for evaluation metrics

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```
### Usage
### Step 1: Data Preprocessing
* Load the raw sensor data from the CSV files.
* Aggregate the data into 30-minute intervals.
* Handle missing values using linear interpolation.
### Step 2: Model Training
* Train the Transformer-based Autoencoder using the unsupervised learning approach.
* The model will learn to reconstruct normal time-series data and detect anomalies based on reconstruction error.
### Step 3: Anomaly Detection
* After training, use the model to calculate the reconstruction error on test data.
* Anomalies are flagged when the reconstruction error exceeds a predefined threshold.
### Step 4: Evaluation
* Assess the model's performance using the reconstruction error distribution and visual inspection of detected anomalies.
### Results
* The trained model provides accurate anomaly detection, with the ability to:

* Identify sudden faults like equipment failures.
* Detect gradual shifts in sensor data, indicating sensor drift or deterioration.
###  Future Work
* Data Augmentation: Improve model robustness by generating synthetic anomaly data.
* Model Enhancements: Experiment with adding more layers or Bidirectional LSTM layers for improved performance.
* Real-Time Deployment: Implement the model in a real-time environment for continuous anomaly detection.
* Cross-Domain Testing: Evaluate the model's performance on different industrial datasets for broader applicability.
