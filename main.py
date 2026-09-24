import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

#_____Teste do Python_____
#print("Python funcionando corretamente!")
#_____Teste das importações_____
#print("Importações realizadas com sucesso.")
#print("OpenCV:", cv2.__version__)

#_____Teste do Tkinter_____
#janela = tk.Tk()
#janela.title("Teste Tkinter")
#janela.geometry("400x200")
#texto = tk.Label(janela, text="Tkinter funcionando!")
#texto.pack(pady=50)
#janela.mainloop()


#_____Teste da webcam_____
#camera = cv2.VideoCapture(0)
#if not camera.isOpened():
# print("Erro: não foi possível acessar a câmera.")
# raise SystemExit
#while True:
# sucesso, frame = camera.read()
# if not sucesso:
# print("Erro ao capturar frame.")
# break
# cv2.imshow("Teste da Webcam", frame)
# if (cv2.waitKey(1) & 0xFF) == ord("q"):
# break
#camera.release()
#cv2.destroyAllWindows()


#_____Teste do YOLO_____
#print("Carregando modelo...")
#modelo = YOLO("yolo11n.pt")
#print("Modelo carregado.")
