# Autism Prediction Model Documentation

## Overview
The Autism Prediction Model is designed to analyze and predict autism spectrum disorder (ASD) tendencies based on questionnaire data and other demographic features. The project follows a robust pipeline to preprocess data, perform transformations, and generate insightful visualizations that assist in understanding trends, patterns, and relationships within the dataset.

## Features
1. **Preprocessing Pipeline**:
   - Handles missing values, outliers, and inconsistent data.
   - Encodes categorical variables into numerical formats.
   - Scales numerical features for better model performance.

2. **Visualization Pipeline**:
   - Generates static and interactive visualizations for numerical and categorical data.
   - Saves plots in both image (`.png`) and interactive HTML formats.

3. **Key Insights**:
   - Identifies correlations between features.
   - Highlights key demographic trends and behavioral patterns.

---

## Workflow

### 1. **Dataset Exploration**
The initial phase involves loading and exploring the dataset using `pandas` to understand its structure, check for missing values, and identify key features.

```python
# Load the dataset
def load_data(file_path):
    """Loads the dataset into a pandas DataFrame."""
    return pd.read_csv(file_path)
```

Key methods:
- `head()`: Displays the first few rows of the dataset.
- `describe()`: Provides statistical summaries for numerical columns.
- `isnull().sum()`: Identifies missing values.

---

### 2. **Data Preprocessing**
The preprocessing pipeline includes:
- **Handling Missing Data**: Missing values are visualized and imputed where necessary.
- **Encoding**: Converts categorical data into numerical representations.
- **Scaling**: Normalizes numerical data using standard scaling techniques.

### 3. **Visualization**
#### Missing Values Heatmap
Visualizes missing data in the dataset:
```python
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
```

#### Numerical Data Distribution
Creates histograms and interactive visualizations for numerical columns:
```python
sns.histplot(df[col], kde=True, color='blue')
```

#### Correlation Heatmap
Analyzes relationships between numerical features:
```python
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
```

#### Scatter Plots
Displays relationships between paired numerical features:
```python
sns.scatterplot(data=df, x=pair[0], y=pair[1], hue=None)
```

#### Pie Charts
Highlights proportions for categorical columns:
```python
plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
```

---

## Deliverables
1. **Preprocessed Dataset**: A cleaned and transformed dataset ready for modeling.
2. **Static Visualizations**:
   - Heatmaps, histograms, scatter plots, and pie charts saved as `.png`.
3. **Interactive Visualizations**:
   - HTML files for interactive exploration of data trends.

## Directory Structure
```
├── Autism_Prediction_Model
│   ├── Data
│   │   └── train_transformed.csv
│   ├── Scripts
│   │   └── main.py
│   ├── Visualizations
│   │   ├── missing_values_heatmap.png
│   │   ├── age_distribution.png
│   │   ├── age_distribution_interactive.html
│   │   ├── correlation_heatmap.png
│   │   └── correlation_heatmap_interactive.html
│   └── README.md
```

---

## How to Run
1. Place your dataset (`train_transformed.csv`) in the `Data` directory.
2. Run the `main.py` script to execute the preprocessing and visualization pipeline:
   ```bash
   python main.py
   ```
3. Access generated visualizations in the `Visualizations` directory.

---

## Example Insights
- **Demographic Trends**: Males in certain age groups show higher ASD tendencies.
- **Feature Correlation**: `age` and `result` exhibit a weak positive correlation.
- **Category Proportions**: Majority of participants did not report prior autism diagnosis.

---

## Tools and Libraries
- **Data Manipulation**: `pandas`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Interactive Analysis**: HTML exports for enhanced accessibility

---

## Conclusion
This project delivers a streamlined approach for analyzing ASD-related data. The combination of preprocessing, transformation, and visualization ensures actionable insights while maintaining clarity and accessibility. The modular design allows for easy adaptation to other datasets or objectives.

_"As deepseek said."_
