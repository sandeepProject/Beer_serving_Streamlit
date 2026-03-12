import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor 
from sklearn.metrics import r2_score

df = pd.read_csv('beer-servings.csv')

df = df.drop(columns=['Unnamed: 0'])

print(df.isnull().sum())
# --- Handling Missing Values ---
# Drop the row where the target variable is missing
df = df.dropna(subset=['total_litres_of_pure_alcohol'])

# Fill missing numerical values with the median
num_cols = ['beer_servings', 'spirit_servings', 'wine_servings']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Final check (Optional: will print in your terminal/logs)
print(df.isnull().sum())

 # Preprocessing: Drop 'country' (too many unique values), Encode 'continent'

X = df.drop(columns=['total_litres_of_pure_alcohol','country'])
y = df['total_litres_of_pure_alcohol']

X_encoded = pd.get_dummies(X, columns=['continent'],  dtype=int)
# Save the column names to align them during prediction
model_columns = list(X_encoded.columns)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Define models and their hyperparameter grids
model_configs = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {'fit_intercept': [True, False]}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        }
    },
    'eX Gradient Boosting': {
        'model':XGBRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200], 
            'learning_rate': [0.01, 0.1]
        }
    }
}

# Loop through the dictionary to tune and evaluate
results = {}
best_score = -1
final_model = None
final_model_name = ""

for name, config in model_configs.items():
    # Run GridSearch
    gs = GridSearchCV(config['model'], config['params'], cv=3, scoring='r2')
    gs.fit(X_train, y_train)
    
    # Evaluate on test data
    y_pred = gs.best_estimator_.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = score
    
    # Update the "Best" model for deployment
    if score > best_score:
        best_score = score
        final_model = gs.best_estimator_
        final_model_name = name

# # Print Final Evaluation
# print("--- Model Performance (R2 Score) ---")
# for name, score in results.items():
#     print(f"{name}: {score:.4f}")

# print(f"\nWinner for Deployment: {final_model_name} with R2: {best_score:.4f}")
# Save Model AND Column List
with open('model_data.pkl', 'wb') as f:
    pickle.dump({
        'model': final_model,
        'best_score': best_score,
        'final_model_name': final_model_name,
        'columns': model_columns,
        'continents': df['continent'].unique().tolist()
    }, f)