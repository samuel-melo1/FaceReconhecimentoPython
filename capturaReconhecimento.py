import face_recognition
import cv2
import os
import imutils

# Pasta onde as imagens de treinamento estão localizadas (rostosPython)
pasta_treinamento = "C:\\Users\\Samuel\\Documents\\rostosPython"

# Carregue as imagens de treinamento e seus rótulos
imagens_treinamento = []
rotulos = []

for arquivo in os.listdir(pasta_treinamento):
    if arquivo.endswith(".png"):  # Certifique-se de que são arquivos de imagem válidos
        caminho_arquivo = os.path.join(pasta_treinamento, arquivo)
        imagem = face_recognition.load_image_file(caminho_arquivo)
        encoding = face_recognition.face_encodings(imagem)
        if len(encoding) > 0:
            imagens_treinamento.append(encoding[0])
            rotulos.append(arquivo.split(".")[0])  # Use o nome do arquivo como rótulo

# Inicialize a câmera (0 é o índice da câmera, pode variar dependendo do sistema)
camera = cv2.VideoCapture(0)

while True:
    # Capture um quadro da câmera
    ret, frame = camera.read()
    frame = imutils.resize(frame, width=800)  # Redimensiona o quadro para uma largura fixa (opcional)

    # Encontre os rostos no quadro
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare o rosto encontrado com os rostos de treinamento
        matches = face_recognition.compare_faces(imagens_treinamento, face_encoding, tolerance=0.4)

        # Adicione um print para a porcentagem de correspondência
        face_distances = face_recognition.face_distance(imagens_treinamento, face_encoding)
        porcentagem_correspondencia = 100 - (min(face_distances) * 100)
        print(f"Porcentagem de correspondência: {porcentagem_correspondencia:.2f}%")

        nome = "Desconhecido"  # Defina um valor padrão

        for i, match in enumerate(matches):
            if match:
                nome = rotulos[i]
                break

        print(f"Pessoa reconhecida: {nome}")

        # Desenhe um retângulo em torno do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

# Libere a câmera
camera.release()
cv2.destroyAllWindows()
