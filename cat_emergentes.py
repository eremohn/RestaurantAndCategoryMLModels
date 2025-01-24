import pickle
import pandas as pd
import streamlit as st
from prophet import Prophet
import os

# Diccionario que mapea las fechas a un valor numérico en meses (p.ej. Junio 2025 = 6, Diciembre 2025 = 12, etc.)
month_mapping = {
    'Junio 2025': 6,
    'Diciembre 2025': 12,
    'Junio 2026': 18,
    'Diciembre 2026': 24,
    'Junio 2027': 30,
    'Diciembre 2027': 36
}

# Función para cargar los modelos entrenados desde la carpeta correspondiente al estado
def load_models(state):
    model_dir = f'model_output_directory_{state.lower()}'
    
    # Verificar que la carpeta exista
    if not os.path.exists(model_dir):
        st.error(f"Ups, no se encontraron modelos para {state}. ¿Estás seguro de que seleccionaste el estado correcto?")
        return None
    
    models = {}
    
    # Leer todos los archivos .pkl en el directorio del estado
    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl"):
            category = filename.replace("_prophet_model.pkl", "")
            model_path = os.path.join(model_dir, filename)
            
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                # Almacenar el modelo en el diccionario
                models[category] = {'model': model, 'data': None}  # Aquí data podría estar, si se desea incluir los datos originales

    return models

# Función para realizar predicciones y calcular tasas de crecimiento
def predict_and_calculate_growth(state, months, growth_rates):
    models = load_models(state)
    if models is None:
        return None

    future_predictions = {}
    
    # Realizar predicciones para cada categoría
    for category, model_data in models.items():
        model = model_data['model']
        
        # Crear DataFrame futuro
        future = model.make_future_dataframe(periods=months, freq='M')
        forecast = model.predict(future)
        
        # Guardar predicción
        future_predictions[category] = forecast

    # Calcular tasas de crecimiento
    growth_results = {}
    for category, forecast in future_predictions.items():
        initial_value = forecast.loc[forecast['ds'] == forecast['ds'].min(), 'yhat'].values[0]
        final_value = forecast.loc[forecast['ds'] == forecast['ds'].max(), 'yhat'].values[0]
        growth_rate = ((final_value - initial_value) / initial_value) * 100
        
        # Aumentar tasa de crecimiento para categorías prioritarias
        if category in ['asian', 'vegan/vegetarian', 'seafood', 'coffee/tea culture', 'mediterranean']:
            growth_rate += 20  # Incremento para categorías prioritarias
        
        growth_results[category] = growth_rate

    # Crear resumen
    growth_summary = pd.DataFrame.from_dict(growth_results, orient='index', columns=['Growth Rate (%)'])
    growth_summary = growth_summary.sort_values(by='Growth Rate (%)', ascending=False)
    return growth_summary

# Configuración de Streamlit
st.title("🔮 Predicción de Categorías Emergentes de Restaurantes 🔮")
st.sidebar.header("🚀 Parámetros de Entrada 🚀")
state = st.sidebar.selectbox(
    "Selecciona un estado 🗺️", 
    ["florida", "california"],
    help="Elige el estado para obtener las predicciones más relevantes."
)

# Selección de fecha en el selectbox
month_selection = st.sidebar.selectbox(
    "¿Hasta qué mes quieres predecir? 📅", 
    options=list(month_mapping.keys()),  # Usamos las fechas como opciones
    index=0,  # El valor predeterminado será 'Junio 2025'
    help="Elige un rango de meses para proyectar las predicciones."
)

# Obtener el valor numérico correspondiente a la fecha seleccionada
months = month_mapping[month_selection]

st.write(f"✨ Predicciones para el estado de *{state.capitalize()}* hasta **{month_selection}** (equivalente a {months} meses) ✨")

# Botón para ejecutar predicción
if st.sidebar.button("¡Hagamos las predicciones! 🎯"):
    growth_rates = {}
    results = predict_and_calculate_growth(state, months, growth_rates)

    if results is not None:
        # Mostrar las 5 categorías principales
        st.write("🔥 **Las 5 categorías que serán tendencia** 🔥")
        top_5 = results.head(5)
        for idx, (category, row) in enumerate(top_5.iterrows(), start=1):
            st.write(f"{idx}. **{category.capitalize()}**  *{row['Growth Rate (%)']:.2f}%*")
    
    # Si no hay resultados
    else:
        st.error("😱 ¡Algo salió mal! No pudimos obtener los resultados. Intenta con otro estado o verifica los modelos.")

