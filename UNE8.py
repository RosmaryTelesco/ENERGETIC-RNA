import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Función para cargar y preparar los datos
def load_and_prepare_data():
    df = pd.read_csv('Plan2024.csv', sep=';', encoding='latin1')
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
    df = df.sort_values('Fecha')
    df['DiaSemana'] = df['Fecha'].dt.dayofweek
    df['MesAno'] = df['Fecha'].dt.month
    
    sequence_length = 7
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[['TOTAL EMPRESA', 'DiaSemana', 'MesAno']].values[i:i+sequence_length])
        y.append(df['TOTAL EMPRESA'].values[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = np.zeros_like(X)
    for i in range(X.shape[2]):
        X_scaled[:,:,i] = scaler_X.fit_transform(X[:,:,i].reshape(-1, 1)).reshape(X.shape[0], X.shape[1])
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    return X_scaled, y_scaled, scaler_X, scaler_y, df

# Función para crear y entrenar el modelo LSTM
def create_and_train_model(X, y):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    
    return model, history

# Función para hacer predicciones
def predict_energy(model, scaler_X, scaler_y, input_data):
    input_scaled = np.zeros_like(input_data)
    for i in range(input_data.shape[2]):
        input_scaled[:,:,i] = scaler_X.transform(input_data[:,:,i].reshape(-1, 1)).reshape(input_data.shape[0], input_data.shape[1])
    prediction_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    return prediction[0][0]

# Interfaz de usuario con Streamlit
st.title('Predictor de Consumo Energético')

# Cargar datos
X_scaled, y_scaled, scaler_X, scaler_y, df = load_and_prepare_data()

# Opción para entrenar el modelo
if st.button('Entrenar Modelo'):
    with st.spinner('Entrenando el modelo...'):
        model, history = create_and_train_model(X_scaled, y_scaled)
        model.save('energy_model_lstm.h5')
    st.success('Modelo LSTM entrenado y guardado como energy_model_lstm.h5')
    
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()
    st.pyplot(fig)

# Entrada de datos del usuario para predicción
prediction_date = st.date_input("Fecha para predicción", datetime.now() + timedelta(days=1))

if st.button('Generar Predicción'):
    if not os.path.exists('energy_model_lstm.h5'):
        st.error("El modelo no ha sido entrenado aún. Por favor, entrena el modelo primero.")
    else:
        model = load_model('energy_model_lstm.h5')
        
        # Preparar datos para la predicción
        last_sequence = df[['TOTAL EMPRESA', 'DiaSemana', 'MesAno']].values[-7:]
        pred_day = prediction_date.weekday()
        pred_month = prediction_date.month
        
        # Hacer la predicción
        input_data = np.array([last_sequence])
        predicted_energy = predict_energy(model, scaler_X, scaler_y, input_data)
        
        st.success(f"Predicción de consumo de energía para {prediction_date}: {predicted_energy:.2f} MWh")
        
        # Visualizar la predicción en el contexto de los datos históricos
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Fecha'], df['TOTAL EMPRESA'], label='Datos históricos')
        ax.scatter(prediction_date, predicted_energy, color='red', s=100, label='Predicción')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Consumo de Energía (MWh)')
        ax.set_title('Predicción de Consumo de Energía')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

st.info('Desarrollado por un Experto en IA')