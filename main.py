import os
import cv2

# Pasta onde as imagens estão localizadas
pasta_imagens = "C:\\Users\\Samuel\\Desktop\\FaceReconhecimento\\FotosRecognition"

# Listar todos os arquivos na pasta
arquivos = os.listdir(pasta_imagens)

# Encontre o arquivo mais recente com base na data de modificação
arquivo_mais_recente = max(arquivos, key=lambda arquivo: os.path.getmtime(os.path.join(pasta_imagens, arquivo)))

# Caminho completo para a imagem mais recente
caminho_imagem_mais_recente = os.path.join(pasta_imagens, arquivo_mais_recente)

# Determinar o próximo número sequencial
numero_sequencial = 0

# Verificar se existem arquivos de rostos anteriores
if os.path.exists("C:\\Users\\Samuel\\Documents\\rostosPython"):
    # Listar todos os arquivos na pasta de destino
    arquivos_rostos = os.listdir("C:\\Users\\Samuel\\Documents\\rostosPython")
    # Encontrar o número sequencial com base nos nomes de arquivos existentes
    numeros_existentes = [int(nome.split("_")[1].split(".")[0]) for nome in arquivos_rostos]
    numero_sequencial = max(numeros_existentes) + 1

# Carregue a imagem mais recente
imagem = cv2.imread(caminho_imagem_mais_recente)

# Carregue o classificador de faces pré-treinado
classificador_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detecte rostos na imagem
rostos = classificador_faces.detectMultiScale(imagem, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Recorte e salve os rostos encontrados
for i, (x, y, w, h) in enumerate(rostos):
    rosto_recortado = imagem[y:y + h, x:x + w]
    nome_arquivo = f"rosto_{numero_sequencial}.png"  # Nomeie os arquivos com o número sequencial
    caminho_salvar = os.path.join("C:\\Users\\Samuel\\Documents\\rostosPython", nome_arquivo)
    cv2.imwrite(caminho_salvar, rosto_recortado)
    print(f"Rosto {numero_sequencial} recortado e salvo em {caminho_salvar}")
    numero_sequencial += 1

print(f"Total de {len(rostos)} rostos encontrados na imagem mais recente.")
