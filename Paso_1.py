import pandas as pd
import numpy as np

# Definimos el número de ejemplos
n_samples = 1000

# Simulamos datos de encuesta
np.random.seed(42)
data = {
    'q1': np.random.randint(1, 6, n_samples),
    'q2': np.random.randint(1, 6, n_samples),
    'q3': np.random.randint(1, 6, n_samples),
    'q4': np.random.randint(1, 6, n_samples),
    'q5': np.random.randint(1, 6, n_samples),
    'q6': np.random.randint(1, 6, n_samples),
    'q7': np.random.randint(1, 6, n_samples),
    'q8': np.random.randint(1, 6, n_samples),
    'q9': np.random.randint(1, 6, n_samples),
    'q10': np.random.randint(1, 6, n_samples),
}

# Convertimos el diccionario en un DataFrame de pandas
df = pd.DataFrame(data)

# Calculamos el nivel de estrés como la suma de las respuestas
df['stress_level'] = df.sum(axis=1)

# Clasificamos el nivel de estrés
def classify_stress_level(score):
    if score <= 20:
        return "muy_bajo"
    elif score <= 30:
        return "bajo"
    elif score <= 40:
        return "medio"
    elif score <= 50:
        return "alto"
    else:
        return "muy_alto"

df['stress_level'] = df['stress_level'].apply(classify_stress_level)

# Simulamos recomendaciones musicales
recommendations = {
    "muy_bajo": "https://www.youtube.com/watch?v=2OEL4P1Rz04",
    "bajo": "https://www.youtube.com/watch?v=5qap5aO4i9A",
    "medio": "https://www.youtube.com/watch?v=DXUAyRRkI6k",
    "alto": "https://www.youtube.com/watch?v=CoUOrLe4vlY",
    "muy_alto": "https://www.youtube.com/watch?v=5yx6BWlEVcY"
}

df['recommended_music'] = df['stress_level'].map(recommendations)

# Guardamos el DataFrame en un archivo CSV
df.to_csv('data/stress_dataset_simulated.csv', index=False)

print("Dataset generado y guardado como 'data/stress_dataset_simulated.csv'.")
