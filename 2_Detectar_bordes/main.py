import numpy as np
import cv2
#import math
import pandas as pd

# Abrir el archivo del video
path = '../recursos/sur_oriente.mp4'
capture = cv2.VideoCapture(path)

# Comprobar que el archivo exista
if not capture.isOpened:
    print('Unable to open: ' + path)
    exit(0)

################################
# MODULO DE ELIMINAR FONDO
################################
backSub = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False)
################################
################################

#Kernel
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))


while True:
    ret, frame = capture.read()
    if frame is None:
        break  

    ##################################
    # CAMBIAR ALTURA Y ANCHO
    ###################################
    height , width , layers =  frame.shape
    resized_frame = cv2.resize(frame, (width//2, height//2)) 
    ###################################

    #-------------------------------------------------
    ##################################
    # ELIMINAR FONDO
    # apply(input,output,learning_rate)
    ##################################
    fgMask = cv2.GaussianBlur(resized_frame, (7, 7), 0)
    fgMask = backSub.apply(fgMask,None, 0.003)
    ##################################

    #-----------------------------------------------------
    ##################################
    # Filtros
    ##################################
    # Remove noise
    opening = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_close)

    # Fill any small holes
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(closing, kernel, iterations=3)

    # threshold
    dilation[dilation < 250] = 0    

    #--------------------------------------------------
    #--------------------------------------------------
    #--------------------------------------------------
    
    ############################################
    #DETECTAR BORDES
    ############################################
    #detectar contornos
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    #obtener rectángulos
    rectangulos = []
    for cn in contours:
        #se obtienen rectágulos
        (x, y, w, h) = cv2.boundingRect(cn)
        #dibuja un rectángulo
        cv2.rectangle(resized_frame, (x,y), (x+w,y+h), (0, 128, 0), 3)
        rectangulos.append((x, y, w, h))
    #-----------------------------------------------------
    
    ######################################
    # CLASIFICAR    
    #######################################
    #pesados = []
    #for rectangulo in rectangulos:
    #    # ancho y alto
    #    #if(rectangulo[2]>=100 and rectangulo[3]>=50):
    #    if(rectangulo[2]>=50 and rectangulo[3]>=20):
    #        pesados.append(rectangulo)
    
    #for pesado in pesados:
    #    #se obtienen rectágulos
    #    (x, y, w, h) = pesado
    #    #dibuja un rectángulo
    #    cv2.rectangle(resized_frame, (x,y), (x+w,y+h), (255, 0, 0), 3)

    ##################################
    # MOSTRAR LA IMAGEN
    ##################################
    cv2.imshow('Frame_mascara_filtros', dilation)
    cv2.imshow('Frame_menor',resized_frame)
    #--------------------------------------------------
    #cerrar el programa al presionar Esc
    keyboard = cv2.waitKey(30)
    ##################################
    ##################################

    if keyboard == 'q' or keyboard == 27:
        break      