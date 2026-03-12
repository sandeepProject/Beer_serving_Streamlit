import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

df = pd.read_csv('beer-servings.csv')
df = df.drop(columns=['Unnamed: 0','country'])
# Load the saved model and column list from your train_model.py
with open('model_data.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    model_score =data['best_score']
    model_name=data['final_model_name']
    model_columns = data['columns']
    continents = data['continents']

st.title("🍺 Beer Servings & Alcohol Predictor")

# Infographics Section
st.header("Global Alcohol Trends")
fig = px.scatter(df, x="beer_servings", y="total_litres_of_pure_alcohol", 
                 color="continent", title="Beer vs Total Pure Alcohol")
st.plotly_chart(fig)
# --- UI Inputs ---
continent = st.selectbox("Continent", continents)
beer = st.number_input("Beer", 0)
spirit = st.number_input("Spirit", 0)
wine = st.number_input("Wine", 0)

if st.button("Predict"):
    # Create a single-row DataFrame from user input
    input_df = pd.DataFrame([[continent, beer, spirit, wine]], 
                            columns=['continent', 'beer_servings', 'spirit_servings', 'wine_servings'])
    
# Apply get_dummies
    input_encoded = pd.get_dummies(input_df, columns=['continent'])  

# ALIGNMENT: Ensure the input has the EXACT same columns as the model
# reindex adds missing columns as NaN, then we fill with 0
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
    
# Predict
    prediction = model.predict(input_final)
    st.success(f"Predicted Alcohol: {prediction[0]:.2f} Litres")  





