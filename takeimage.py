import cv2
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import imutils
import time
import re
import requests

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})

image_directory = 'C:\\Users\\Samuel\\Desktop\\FaceReconhecimento\\FotosRecognition'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
known_face_encodings = []
known_face_names = []
camera = cv2.VideoCapture(0)

def extrair_cpf(nome_imagem):
    cpf_pattern = r'\d{3}\.\d{3}\.\d{3}-\d{2}'  # Padrão de CPF com pontos e hífen
    match = re.search(cpf_pattern, nome_imagem)
    if match:
        return match.group()
    else:
        return None
def get_next_image_path(cpf):
    existing_files = os.listdir(image_directory)

    i = 0
    while True:
        image_name = f'{cpf}_{i}.jpg'
        if image_name not in existing_files:
            return os.path.join(image_directory, image_name)
        i += 1

@app.route('/api/pessoa/tirar_foto', methods=['POST'])
def tirar_foto():
    data = request.get_json()
    cpf = data['cpf']

    image_path = get_next_image_path(cpf)

    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()

    camera.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    cv2.imwrite(image_path, frame)

    return jsonify({'message': 'Foto tirada com sucesso', 'image_path': image_path})



# Carregar as imagens da pasta e calcular os códigos de reconhecimento facial
for file_name in os.listdir(image_directory):
    if file_name.endswith(".jpg"):
        image_path = os.path.join(image_directory, file_name)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            print(f"Não foi possível encontrar uma face na imagem: {file_name}")
            continue

        # Suponha uma única face por imagem (você pode adaptar para várias faces por imagem)
        known_face_encodings.append(face_encodings[0])
        known_face_names.append(os.path.splitext(file_name)[0])  # E

@app.route('/api/pessoa/reconhecimento_facial', methods=['POST'])
def reconhecimento_facial():
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    camera.release()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(frame)

    if not face_locations:
        return jsonify({'message': 'Nenhuma face encontrada'})

    face_encodings = face_recognition.face_encodings(frame, face_locations)

    results = face_recognition.compare_faces(known_face_encodings, face_encodings[0], tolerance=0.5)
    name = "Desconhecido"
    cpf = ""

    if True in results:
        index = results.index(True)
        name = known_face_names[index]
        cpf = extrair_cpf(name)
    if cpf:
        # Enviar o CPF para o serviço Java
        response = requests.post('http://localhost:8080/api/log/registrar_log', json={'cpf': cpf})
        if response.status_code == 200:
            return jsonify({'message': 'Reconhecimento facial realizado', 'name': cpf})
        else:
            return jsonify({'message': 'Erro ao registrar log', 'name': cpf}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Execute o servidor Flask na porta 5000
