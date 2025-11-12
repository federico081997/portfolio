# Australia Wildfires Dashboard

An interactive **data visualization dashboard** built with [Plotly Dash](https://dash.plotly.com/) to explore wildfire activity across Australian regions and years. The project demonstrates modular app architecture, clean data processing and effective visual storytelling for environmental analytics.

## Overview

This dashboard enables users to analyze **wildfire patterns** in Australia through two key metrics:

1. **Estimated Fire Area** — average monthly burned area (km²).
2. **Pixel Count for Presumed Vegetation Fires** — proxy for fire frequency and spatial distribution.

Users can dynamically filter the data by **region** and **year**, and view the results via:
- A **donut chart** (average estimated fire area).
- A **bar chart** (average monthly pixel count).

## Features

- Interactive region and year filters.
- Real-time chart updates through Dash callbacks.
- Modular code structure (layout, logic, figures, callbacks).
- Clean, responsive layout using HTML and CSS properties.
- Data-driven annotations and legends for better interpretability.

## Project Structure

Australia-Wildfires-Dashboard/
│
├── data/
│ └── Historical_Wildfires.csv
│
├── src/
│ ├── init.py
│ ├── app.py
│ ├── callbacks.py
│ ├── constants.py
│ ├── data_io.py
│ ├── figures.py
│ ├── layout.py
│ └── logic.py
│
├── README.md
└── requirements.txt


### Key modules
- **`app.py`** — Entry point; initializes Dash app, layout, and callbacks.
- **`layout.py`** — Builds Dash UI layout (controls and charts).
- **`callbacks.py`** — Handles interactivity and chart updates.
- **`logic.py`** — Data filtering and aggregation.
- **`figures.py`** — Chart construction using Plotly Express.
- **`data_io.py`** — Data loading utilities.
- **`constants.py`** — Region and year constants.

## 🧮 Data

- **File:** `Historical_Wildfires.csv`
- **Source:** Processed dataset containing monthly wildfire statistics by Australian region.
- **Key columns:**
  - `Region`, `Year`, `Month_name`, `Estimated_fire_area`, `Count`

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/federico081997/Australia-Wildfires-Dashboard.git
cd Australia-Wildfires-Dashboard
```

### 2. Create a Python virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app from the project root
```bash
python -m src.app
```



