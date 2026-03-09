# AutoML Pipeline for Tabular Data with MLflow

An automated machine learning (AutoML) pipeline designed to streamline the process of model selection, hyperparameter tuning, and experiment tracking for tabular datasets. This project demonstrates an end-to-end workflow using the Titanic dataset, incorporating multiple machine learning algorithms and MLflow for comprehensive experiment management.

## Features

- **Automated Model Selection**: Supports multiple algorithms including Random Forest, Logistic Regression, and XGBoost.
- **Hyperparameter Tuning**: Utilizes Grid Search Cross-Validation for optimal parameter selection.
- **Experiment Tracking**: Integrates MLflow to log parameters, metrics, and artifacts for each experiment run.
- **Model Evaluation**: Comprehensive evaluation using accuracy, F1-score, ROC-AUC, and detailed classification reports.
- **Model Persistence**: Saves the best-performing model for future predictions.
- **Interactive Notebook**: Includes a Jupyter notebook for exploratory data analysis (EDA) on the Titanic dataset.
- **Easy Testing**: Provides a script to load and test the trained model with sample data.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Jashan-jj/AutoML-Pipeline-for-Tabular-Data-with-MLflow.git
   cd AutoML-Pipeline-for-Tabular-Data-with-MLflow
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7+ installed. Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up MLflow Tracking Server** (Optional):
   For advanced tracking, start an MLflow server:
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```
   The pipeline is configured to use `http://127.0.0.1:5000` as the tracking URI. You can modify this in the script if needed.

## Usage

### Running the AutoML Pipeline

Execute the main pipeline script to train models and select the best one:
```bash
python scripts/AutoML_pipline.py
```

This script will:
- Load the Titanic dataset.
- Preprocess the data.
- Train and tune multiple models.
- Log experiments to MLflow.
- Save the best model as `best_model.pkl`.

### Testing the Model

Use the test script to make predictions with the trained model:
```bash
python scripts/test_model.py
```

This will load the saved model and provide a prediction for a sample passenger data.

### Viewing Experiments in MLflow

After running the pipeline, view the logged experiments:
```bash
mlflow ui
```
Open your browser to `http://127.0.0.1:5000` to explore runs, compare models, and view artifacts.

### Exploratory Data Analysis

Open the Jupyter notebook for detailed EDA:
```bash
jupyter notebook notebooks/titanic.ipynb
```
This notebook includes data visualization, preprocessing steps, and insights into the Titanic dataset.

## Dataset

The project uses the cleaned Titanic dataset (`datasets/titanic_cleaned.csv`), which includes features such as passenger class, sex, age, fare, and embarkation point, with the target variable being survival status.

## Technologies Used

- **Python**: Core programming language.
- **Scikit-learn**: For machine learning algorithms and evaluation.
- **XGBoost**: Gradient boosting framework.
- **MLflow**: Experiment tracking and model management.
- **Pandas & NumPy**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization (in the notebook).
- **Jupyter Notebook**: Interactive data exploration.

## Project Structure

```
AutoML-Pipeline-for-Tabular-Data-with-MLflow/
├── datasets/
│   └── titanic_cleaned.csv          # Cleaned Titanic dataset
├── mlruns/                          # MLflow experiment tracking directory
├── notebooks/
│   └── titanic.ipynb                # EDA notebook
├── scripts/
│   ├── AutoML_pipline.py            # Main AutoML pipeline script
│   └── test_model.py                # Model testing script
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
