import cv2 #OpenC

imagem = cv2.imread("brasil.jpg")
cv2.imshow("Brasil", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

detector_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imshow("Brasil", imagem_cinza)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

deteccoes = detector_face.detectMultiScale(imagem_cinza)
len(deteccoes)

for (x, y, l, a) in deteccoes:
    cv2.rectangle(imagem, (x,y), (x + l, y + a), (0, 255, 0), 2)
cv2.imshow("Brasil", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)    