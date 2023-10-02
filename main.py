import os
import cv2

# Pasta onde as imagens estão localizadas
pasta_imagens = "C:\\Users\\Samuel\\Documents\\pythonTeste"

# Listar todos os arquivos na pasta
arquivos = os.listdir(pasta_imagens)

# Encontre o arquivo mais recente com base na data de modificação
arquivo_mais_recente = max(arquivos, key=lambda arquivo: os.path.getmtime(os.path.join(pasta_imagens, arquivo)))

# Caminho completo para a imagem mais recente
caminho_imagem_mais_recente = os.path.join(pasta_imagens, arquivo_mais_recente)

# Carregue a imagem mais recente
imagem = cv2.imread(caminho_imagem_mais_recente)

# Carregue o classificador de faces pré-treinado
classificador_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detecte rostos na imagem
rostos = classificador_faces.detectMultiScale(imagem, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Recorte e salve os rostos encontrados
for i, (x, y, w, h) in enumerate(rostos):
    rosto_recortado = imagem[y:y + h, x:x + w]
    nome_arquivo = f"rosto_{i}.jpg"  # Nomeie os arquivos de acordo com a posição do rosto
    caminho_salvar = os.path.join("C:\\Users\\Samuel\\Documents\\rostosPython", nome_arquivo)
    cv2.imwrite(caminho_salvar, rosto_recortado)
    print(f"Rosto {i} recortado e salvo em {caminho_salvar}")

print(f"Total de {len(rostos)} rostos encontrados na imagem mais recente.")
