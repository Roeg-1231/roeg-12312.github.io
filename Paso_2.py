import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Paso 1: Carga del Dataset
df = pd.read_csv('data/stress_dataset_simulated.csv')

# Paso 2: Preprocesamiento de Datos
# Dividimos las características y las etiquetas
X = df.drop(['stress_level', 'recommended_music'], axis=1)
y = df['stress_level']

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizamos las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Paso 3: Entrenamiento del Modelo
# Entrenamos un modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Paso 4: Evaluación del Modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Paso 5: Guardar el Modelo y el Scaler
joblib.dump(model, 'models/stress_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Modelo y scaler guardados como 'models/stress_model.pkl' y 'scaler.pkl'.")
