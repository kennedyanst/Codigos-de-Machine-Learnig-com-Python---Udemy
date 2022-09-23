from PIL import Image
import numpy as np 
import cv2
import os

#TREINAMENTO
def dados_imagem():
    caminhos = [os.path.join("train", f) for f in os.listdir("train")] #Vai juntar a pasta com o f, e o f percorre todas as imagens
    faces = []
    ids = []
    for caminho in caminhos:
        imagem = Image.open(caminho).convert("L") #L = Usar um canal da imagem (Escala de cinza)
        imagem_np = np.array(imagem, "uint8")
        id = int(os.path.split(caminho)[1].split(".")[0].replace("subject", ""))
        ids.append(id)
        faces.append(imagem_np)
    return np.array(ids), faces


ids, faces = dados_imagem()
print(ids)
print(faces)

lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(faces, ids)
lbph.write("classificadorLBPH.yml") #Arquivo treinado

#CLASSIFICAÇÃO
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("classificadorLBPH.yml")

imagem_teste = "test\subject10.sad.gif"
imagem = Image.open(imagem_teste).convert("L")
imagem_np = np.array(imagem, "uint8")
print(imagem_np)

idcorreto = int(os.path.split(imagem_teste)[1].split(".")[0].replace("subject", ""))
idprevisto, _ = reconhecedor.predict(imagem_np)


cv2.putText(imagem_np, "P: " + str(idprevisto), (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.putText(imagem_np, "C: " + str(idcorreto), (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.imshow("img", imagem_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)    



#imagem = cv2.imread("imagem"img)
#cv2.imshow(imagem)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.waitKey(1)    