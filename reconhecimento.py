import cv2
import os

# Pasta onde as imagens serão salvas
pasta_destino = "C:\\Users\\Samuel\\Documents\\reconhece"
if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

# Inicialize a câmera (0 é o índice da câmera, pode variar dependendo do sistema)
camera = cv2.VideoCapture(0)

# Carregue o classificador de rosto pré-treinado
classificador_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Encontre o número sequencial baseado nas imagens existentes na pasta
contador_sequencial = 0
imagens_na_pasta = os.listdir(pasta_destino)

if imagens_na_pasta:
    numeros_sequenciais = [int(imagem.split("_")[1].split(".")[0]) for imagem in imagens_na_pasta]
    contador_sequencial = max(numeros_sequenciais) + 1

capturou_foto = False

while not capturou_foto:
    # Capture um quadro da câmera
    ret, frame = camera.read()

    # Converta o quadro para escala de cinza
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte rostos no quadro
    rostos = classificador_faces.detectMultiScale(frame_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in rostos:
        # Recorte o rosto
        rosto_recortado = frame[y:y + h, x:x + w]

        # Gere o nome do arquivo para a foto com base no contador sequencial
        nome_arquivo = f"rosto_{contador_sequencial}.png"

        # Salve a imagem do rosto na pasta de destino
        caminho_salvar = os.path.join(pasta_destino, nome_arquivo)
        cv2.imwrite(caminho_salvar, rosto_recortado)
        print(f"Rosto recortado e salvo em {caminho_salvar}")

        contador_sequencial += 1
        capturou_foto = True

# Libere a câmera
camera.release()
