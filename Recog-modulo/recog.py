#import de pacotes
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# abrir câmera (pode usar CAP_DSHOW para mais controle no Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 60)  # alterar FPS

model = YOLO('yolov8l.pt')  #escolha do modelo,
#podemos escolher M ou L pra ter mais precisão

frame_cont = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_cont += 1
    if frame_cont % 300 == 0:  # Processa a cada 300 frames            
    #predição e ajuste do arquivo
        results = model(frame, classes=[0], conf=0.4, device="cpu")

        # #Mostra da imagem
        r = results[0]
        annotated = r.plot()[:, :, ::-1]  # converte BGR -> RGB
        plt.figure(figsize=(10,10))
        plt.axis('off')

        cv2.imshow("Pessoas", annotated) 
    # Contas as pessoas detectadas
        boxes = r.boxes
        print("Número de pessoas detectadas:", boxes.shape[0])
    else:
        cv2.imshow("Pessoas", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            break

cap.release()
cv2.destroyAllWindows()