#import de pacotes
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import shutil

# abrir câmera (pode usar CAP_DSHOW para mais controle no Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 15)  # pedir 60fps

print("FPS    :", cap.get(cv2.CAP_PROP_FPS))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

#     # #Carregando o modelo
    model = YOLO('yolov8n.pt')  #escolha do modelo,
#podemos escolher M ou L pra ter mais precissão

#predição e ajuste do arquivo
    results = model.predict(
        source=frame ,  # Usa o nome exato do arquivo
        classes=[0],         # Apenas pessoas
        conf=0.5,            # Confiança mínima
        device="cpu",        # GPU ou CPU
        save=True            # Salva imagem anotada
    )

    # #Mostra da imagem no Colab
    r = results[0]               # resultado da primeira imagem
    annotated = r.plot()[:, :, ::-1]  # converte BGR -> RGB
    plt.figure(figsize=(10,10))
    plt.imshow(annotated)
    plt.axis('off')
    plt.show()

# Contas as pessoas detectadas
    boxes = r.boxes
    print("Número de pessoas detectadas:", boxes.shape[0])

cap.release()
cv2.destroyAllWindows()