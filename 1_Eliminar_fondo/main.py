import numpy as np
import cv2
import math
import pandas as pd

# Abrir el archivo del video
path = '../recursos/sur_oriente.mp4'
capture = cv2.VideoCapture(path)

# Comprobar que el archivo exista
if not capture.isOpened:
    print('Unable to open: ' + path)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break  

    cv2.imshow('Frame', frame)

    #cerrar el programa
    keyboard = cv2.waitKey(30)

    if keyboard == 'q' or keyboard == 27:
        break      