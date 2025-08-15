# ML Tutorial 

This project demonstrates **data preprocessing**, **linear regression**, and **cross-validation** using Python and Scikit-learn.  
It uses:
- A `tip.csv` dataset for preprocessing
- Example datasets for ice cream and coffee sales

## 📂 Project Structure
ML Tutorial/
│-- tip.csv # Tips dataset
│-- tutorial_script.py # Main script with all steps & plots
│-- README.md # Project documentation

## 🧠 Features
1. **Data Preprocessing**
   - Handling missing values using `SimpleImputer`
   - Scaling numerical features using `StandardScaler`
   - Merging numeric & non-numeric data
   - Dropping missing rows

2. **Linear Regression**
   - Predicting ice cream sales from temperature
   - Predicting coffee sales from temperature

3. **Cross-Validation**
   - Using `cross_val_score` to evaluate model performance
   - R² score calculation

4. **Data Visualization**
   - Scatter plots with regression lines
   - Comparison graph for ice cream vs coffee sales

## 📊 Example Outputs
- **Tip vs Total Bill** – Scatter plot & regression line  
- **Ice Cream Sales** – Scatter plot & regression line  
- **Coffee Sales** – Scatter plot, regression line & cross-validation score  
- **Combined Trends** – Ice cream vs coffee sales

## ⚙️ Installation & Setup
1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/ml-tutorial.git
   cd ml-tutorial
## Install dependencies
pip install pandas numpy matplotlib scikit-learn

## Run the scirpt
python tutorial_script.py
