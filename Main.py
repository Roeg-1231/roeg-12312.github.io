from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import googleapiclient.discovery

app = Flask(__name__)

# Cargar el modelo de Machine Learning y el scaler
model = joblib.load('models/stress_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# YouTube API
def get_youtube_service():
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyDhTgMPE9MlSDw6PTFKJRmX-iftNx-T5oo"
    
    return googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

def search_music_on_youtube(query):
    youtube = get_youtube_service()
    request = youtube.search().list(
        part="snippet",
        maxResults=5,
        q=query,
        type="video"
    )
    response = request.execute()
    videos = []
    for item in response['items']:
        video_info = {
            'title': item['snippet']['title'],
            'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
            'thumbnail': item['snippet']['thumbnails']['default']['url']
        }
        videos.append(video_info)
    return videos

# Preguntas del formulario
questions = [
    "En una escala del 1 al 5, ¿cómo describirías tu estado emocional actual?",
    "En una escala del 1 al 5, ¿cuánto estrés has sentido en la última semana?",
    "En una escala del 1 al 5, ¿cómo calificarías la calidad de tu sueño?",
    "En una escala del 1 al 5, ¿cuánto ejercicio físico has hecho en la última semana?",
    "En una escala del 1 al 5, ¿cuánto tiempo libre has tenido para ti mismo/a?",
    "En una escala del 1 al 5, ¿cómo calificarías tu nivel de energía diaria?",
    "En una escala del 1 al 5, ¿cuán abrumado/a te has sentido en tu trabajo o estudios?",
    "En una escala del 1 al 5, ¿cuánto apoyo emocional has recibido de amigos o familiares?",
    "En una escala del 1 al 5, ¿cuán a menudo has tenido pensamientos negativos?",
    "En una escala del 1 al 5, ¿cómo calificarías tu nivel de satisfacción con la vida en general?"
]

@app.route('/')
def home():
    return render_template('index.html', questions=questions)

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)
    
    stress_level = prediction[0]
    query = "relaxing music"
    if stress_level == "muy_bajo":
        query = "very relaxing music"
    elif stress_level == "bajo":
        query = "calm music"
    elif stress_level == "medio":
        query = "meditation music"
    elif stress_level == "alto":
        query = "stress relief music"
    elif stress_level == "muy_alto":
        query = "deep relaxation music"
    
    music_videos = search_music_on_youtube(query)
    
    return render_template(
        'index.html', 
        prediction_text=f'Nivel de Estrés: {stress_level}', 
        music_videos=music_videos, 
        questions=questions,
        details=f"Su nivel de estrés ha sido clasificado como {stress_level}. Aquí hay algunas recomendaciones musicales para ayudarle a relajarse."
    )

if __name__ == "__main__":
    app.run(debug=True)
