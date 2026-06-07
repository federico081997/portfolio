# Analytics & Machine Learning Workspace

A modular Streamlit application for end-to-end tabular data analysis, preprocessing, visualization, geospatial exploration, classical machine learning, and configurable deep learning workflows.

The project is designed as a portfolio-grade analytics workspace that separates user-interface components from reusable data-processing and modeling logic. It supports dataset ingestion, stateful ETL operations, interactive exploratory analysis, geospatial mapping, supervised learning pipelines, and custom neural network construction.

---

## Overview

The application provides a structured workflow for moving from raw data to trained predictive models:

1. **Data Ingestion**
2. **ETL & Data Engineering**
3. **Exploratory Data Analysis**
4. **Geospatial Mapping**
5. **Predictive Modeling**
6. **Deep Learning**

The application is built with Streamlit and uses pandas, NumPy, SciPy, Scikit-Learn, Plotly, Folium, and TensorFlow/Keras.

---

## Key Features

### Data Ingestion

- Upload local CSV files.
- Load CSV datasets from remote URLs.
- Cache loaded datasets across Streamlit reruns.
- Display user-friendly errors for empty, invalid, or malformed files.
- Store datasets in Streamlit session state for downstream modules.

### ETL and Data Engineering

- Remove selected columns.
- Remove duplicate rows.
- Drop rows with missing values in critical fields.
- Parse datetime columns using configurable formats.
- Extract temporal features such as:
  - year
  - quarter
  - month
  - day
  - hour
  - weekday
  - weekend flags
  - month-start and month-end flags
- Apply cyclical sine/cosine encodings.
- Standardize text casing, spacing, punctuation, and numeric content.
- Analyze and reduce categorical cardinality.
- Downcast numerical types and convert suitable string columns to categorical types.
- Impute numerical and categorical missing values.
- Detect and treat outliers using:
  - IQR
  - Z-score
  - percentile thresholds
- Analyze and transform skewed numerical features.
- Scale numerical features.
- Encode categorical features.
- Maintain a transformation history with undo and reset controls.

### Exploratory Data Analysis

- Numerical histograms with marginal box plots.
- Categorical frequency bar charts.
- Box and violin plots.
- Scatter plots with:
  - color grouping
  - marker sizing
  - OLS trendlines
  - LOWESS smoothing
  - marginal distributions
- Time-series charts with rolling moving averages.
- Pearson and Spearman correlation heatmaps.
- Multivariate scatter matrices.
- Hierarchical sunburst charts.
- Persistent visualization state across Streamlit reruns.

### Geospatial Analytics

- Automatic detection of likely latitude and longitude columns.
- Configurable coordinate selection.
- Browser-safe random downsampling.
- Folium marker-cluster maps.
- Configurable marker popups.
- Category-based marker colors.
- Weighted and unweighted density heatmaps.
- Multiple base-map styles.
- Animated Plotly spatial maps.
- Frame-count protection for large animations.
- Persistent map state across user interactions.

### Classical Machine Learning

Supported classification models:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine
- K-Nearest Neighbors Classifier

Supported regression models:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor
- K-Nearest Neighbors Regressor

Pipeline features:

- Numerical and categorical preprocessing with `ColumnTransformer`.
- Leakage-aware training pipelines.
- Missing-value imputation.
- Outlier clipping or nullification inside the pipeline.
- Optional row removal before train/test splitting.
- Skewness correction.
- Numerical scaling.
- One-hot or ordinal encoding.
- Class weighting for supported classifiers.
- Configurable train/test split.
- Dynamic model hyperparameters.
- Classification metrics:
  - accuracy
  - weighted precision
  - weighted recall
  - weighted F1 score
  - confusion matrix
- Regression metrics:
  - RMSE
  - MAE
  - R²
- Feature-importance plots for supported estimators.
- Test-set diagnostic visualizations.

### Deep Learning

- Build neural networks layer by layer.
- Supported hidden-layer types:
  - Dense
  - Dropout
  - Batch Normalization
- Supported activation functions:
  - ReLU
  - ELU
  - Tanh
  - Sigmoid
  - Swish
- Supported tasks:
  - binary classification
  - multi-class classification
  - regression
- Supported optimizers:
  - Adam
  - Nadam
  - SGD with momentum
  - RMSprop
- Configurable:
  - learning rate
  - batch size
  - maximum epochs
  - validation split
  - early stopping
  - learning-rate decay
- Live Streamlit training telemetry.
- Training and validation learning curves.
- Classification confusion matrices.
- Regression actual-versus-predicted diagnostics.
- Architecture and parameter summaries.
- CSV export of training history.
- JSON export of the compiled architecture profile.

---

## Project Architecture

```text
Analytics-and-Machine-Learning-Workspace/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   ├── config.toml
│   ├── secrets.toml
│
├── components/
│   ├── __init__.py
│   ├── auth.py
│   ├── sidebar.py
│   ├── ingestion_ui.py
│   ├── cleaning_ui.py
│   ├── eda_ui.py
│   ├── geospatial_ui.py
│   ├── ml_ui.py
│   └── dl_ui.py
│
└── core/
    ├── __init__.py
    ├── data_loader.py
    ├── preprocessor.py
    ├── eda_visualizations.py
    ├── geospatial_visualizations.py
    ├── ml_engine.py
    ├── ml_visualizations.py
    └── dl_engine.py
```

### Separation of Responsibilities

The project follows a modular architecture:

- `components/` contains Streamlit interface and routing logic.
- `core/` contains reusable data-processing, visualization, machine learning, and deep learning logic.
- `app.py` configures the application, handles authentication, initializes session state, and routes users to the selected module.
- `.streamlit/` contains theme and local secret configuration.

This separation keeps the UI layer independent from the analytical backend and makes individual functions easier to test, reuse, and maintain.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Analytics-and-Machine-Learning-Workspace
```

### 2. Create a virtual environment

Python 3.12 is recommended.

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Streamlit Configuration

Create the following file:

```text
.streamlit/config.toml
```

Example:

```toml
[theme]
primaryColor = "#4488F6"
backgroundColor = "#F8FAFC"
secondaryBackgroundColor = "#F1F5F9"
textColor = "#0F172A"
font = "sans serif"
```

---

## Authentication Configuration

Create:

```text
.streamlit/secrets.toml
```

Example:

```toml
[credentials]
username = "your_username"
password = "your_secure_password"
```

Do not commit real credentials to version control.

Add this entry to `.gitignore`:

```gitignore
.streamlit/secrets.toml
```

A safe template can be committed as:

```text
.streamlit/secrets.example.toml
```

```toml
[credentials]
username = "your_username_here"
password = "your_secure_password_here"
```

---

## Running the Application

Start the Streamlit server from the project root:

```bash
streamlit run app.py
```

The terminal will display the local application URL, typically:

```text
http://localhost:8501
```

---

## Application Workflow

### 1. Authenticate

Enter the credentials stored in `.streamlit/secrets.toml`.

### 2. Load a Dataset

Use the Data Ingestion page to:

- upload a CSV file, or
- provide a direct URL to a CSV file.

The dataset is stored in:

```python
st.session_state["raw_data"]
```

### 3. Clean and Transform the Data

Use the ETL module to apply structural and analytical transformations.

The latest dataset version is stored in:

```python
st.session_state["cleaned_data"]
```

Transformation versions are stored in:

```python
st.session_state["data_history"]
```

### 4. Explore the Dataset

Use the EDA Studio to inspect distributions, relationships, time-series behavior, multicollinearity, and hierarchical segments.

### 5. Explore Spatial Data

Select latitude and longitude columns and generate clustered maps, density heatmaps, or animated spatial visualizations.

### 6. Train Classical Machine Learning Models

Select the target, predictors, preprocessing configuration, estimator, and hyperparameters. The trained pipeline and evaluation results are stored in session state.

### 7. Train Deep Learning Models

Configure preprocessing, create a neural network architecture, select optimizer settings, and monitor training progress in real time.

---

## Session-State Design

The application uses Streamlit session state to preserve data and outputs between reruns.

Important keys include:

```text
raw_data
cleaned_data
data_history
saved_distribution_plot
saved_variance_plot
saved_relationships_plot
saved_time_series_plot
saved_heatmap
saved_matrix
saved_sunburst_plot
saved_cluster_map
saved_heat_map
saved_anim_map
ml_results
ml_task
trained_model_name
dl_layers
dl_results
dl_task
password_correct
```

This approach allows generated charts, maps, training results, and preprocessing history to remain visible while users interact with other controls.

---

## Preprocessing and Leakage Considerations

The project separates structural transformations from analytical transformations.

Structural operations such as column removal, duplicate removal, text standardization, and datetime extraction can be applied before model training.

Operations that estimate distributional statistics should generally be fitted only on the training set. The machine learning engine therefore places imputation, scaling, encoding, outlier treatment, and skewness transformations inside Scikit-Learn pipelines where possible.

### Important Deep Learning Evaluation Note

The current deep learning engine uses the held-out split as both:

- validation data during training, and
- final evaluation data.

Because early stopping and learning-rate scheduling monitor this split, final test metrics are not fully independent. A stricter production workflow should create separate training, validation, and test subsets.

---

## Known Limitations

- The application is designed primarily for tabular datasets.
- Very large datasets may require additional backend optimization.
- Folium maps can become expensive to render, so the UI applies configurable downsampling.
- Plotly animations with many unique frames may consume substantial browser memory.
- Feature importance is unavailable for estimators without native coefficients or feature-importance attributes.
- Label encoding introduces integer category codes but does not imply ordinal meaning.
- Forward-fill categorical imputation depends on row ordering.
- Deep learning currently supports fully connected sequential architectures rather than convolutional, recurrent, transformer, or graph neural networks.
- Authentication uses application-level credentials stored in Streamlit secrets and is not a substitute for a full identity-management platform.
- The current deep learning workflow should use a separate validation set before being treated as a strict production evaluation pipeline.
- Selecting `Drop Rows` for outlier handling in the deep learning interface requires corresponding pre-split row-removal logic in the deep learning engine.

---

## Deep Learning Navigation Integration

Ensure the main application navigation includes the Deep Learning module.

Example sidebar option:

```python
"6. Deep Learning"
```

Example route in `app.py`:

```python
elif active_phase == "6. Deep Learning":
    render_dl_module()
```

Required import:

```python
from components.dl_ui import render_dl_module
```

---

## Security Notes

- Never commit `.streamlit/secrets.toml`.
- Do not use default credentials such as `admin` and `password`.
- Rotate exposed credentials immediately.
- Validate remote dataset URLs before using the application in a public deployment.
- Avoid displaying sensitive columns in map popups or exported artifacts.
- For production deployment, use a proper authentication and authorization provider.

---

## Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyd
*.so

# Virtual environments
.venv/
venv/
env/

# Streamlit secrets
.streamlit/secrets.toml

# IDE settings
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/

# Operating system files
.DS_Store
Thumbs.db

# Logs and generated artifacts
*.log
training_logs.csv
model_architecture.json

# TensorFlow and model artifacts
*.keras
*.h5
saved_model/
checkpoints/
```

---

## Technology Stack

| Area | Technology |
|---|---|
| Application framework | Streamlit |
| Data manipulation | pandas |
| Numerical computing | NumPy |
| Statistical transformations | SciPy |
| Machine learning | Scikit-Learn |
| Deep learning | TensorFlow / Keras |
| Interactive charts | Plotly |
| Geospatial maps | Folium / Leaflet |
| Statistical trendlines | statsmodels |
| Configuration | TOML |

---

## Development Principles

The project follows several practical engineering principles:

- modular separation between UI and backend logic
- reusable transformation and visualization functions
- defensive dataset validation
- session-state persistence
- consistent function documentation
- explicit train/test processing
- user-facing error messages
- browser-memory protection for heavy visualizations
- reproducible random seeds where applicable

---

## Future Improvements

Potential extensions include:

- separate train, validation, and test splits for deep learning
- cross-validation and automated model comparison
- hyperparameter search with GridSearchCV, RandomizedSearchCV, or Optuna
- model persistence and reload support
- downloadable fitted Scikit-Learn pipelines
- TensorFlow model export
- probability calibration
- ROC and precision-recall curves
- multiclass classification reports
- SHAP-based explainability
- permutation feature importance
- data-quality reports
- automated schema validation
- support for Excel, Parquet, JSON, and database sources
- time-series forecasting workflows
- convolutional and recurrent neural network builders
- experiment tracking
- unit and integration tests
- CI/CD validation
- containerized deployment with Docker
- production authentication and role-based access control

---

## Testing Recommendations

Suggested tests include:

- CSV loader behavior for valid, empty, and malformed files
- preprocessing functions with missing or invalid columns
- datetime parsing failures
- text-cleaning behavior with null values
- outlier-bound calculations
- cardinality-reduction strategies
- memory downcasting
- custom Scikit-Learn transformer compatibility
- preprocessing feature-name extraction
- model construction for every supported estimator
- classification and regression training workflows
- deep learning output-layer selection
- live callback history accumulation
- Plotly figure generation
- coordinate-column detection
- session-state initialization and reset behavior

---

## Disclaimer

This project is intended for educational, portfolio, and exploratory analytics use. Model outputs should not be used as the sole basis for medical, financial, legal, safety-critical, or other high-impact decisions without appropriate validation, governance, and domain-expert review.

---

## Author

**Federico Mazzanti**

Computational engineer transitioning into data science, machine learning, AI, and research engineering.

Background includes numerical methods, scientific computing, mathematical modeling, C++, Python, finite-volume methods, and high-performance simulation.

---

## License

Add the license appropriate for the repository, for example:

```text
MIT License
```

Create a separate `LICENSE` file before publishing the project under a specific license.
