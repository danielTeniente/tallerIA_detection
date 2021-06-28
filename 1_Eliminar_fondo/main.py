import numpy as np
import cv2
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
# 2
#backSub = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False)
################################
################################

#Kernel
# 4
#menores a 5,5 produce muchos bordes
#kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


while True:
    ret, frame = capture.read()
    if frame is None:
        break  

    ##################################
    # CAMBIAR ALTURA Y ANCHO
    ###################################
    # 1
    #height , width , layers =  frame.shape
    #resized_frame = cv2.resize(frame, (width//3, height//3)) 
    ###################################

    #-------------------------------------------------
    ##################################
    # ELIMINAR FONDO
    # apply(input,output,learning_rate)
    ##################################
    # 2
    #back_mask = backSub.apply(resized_frame,None, 0.003)
    # 3
    #fgMask = cv2.GaussianBlur(resized_frame, (5, 5), 0)
    #fgMask = backSub.apply(fgMask,None, 0.003)
    ##################################

    #-----------------------------------------------------
    ##################################
    # Filtros
    ##################################
    
    # 4
    # Remove noise
    #opening = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_close)

    # Fill any small holes
    #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Dilate to merge adjacent blobs
    #dilation = cv2.dilate(closing, kernel, iterations=3)

    # threshold
    #dilation[dilation < 250] = 0    

    #-----------------------------------------------------
    ##################################
    # MOSTRAR LA IMAGEN
    ##################################
    cv2.imshow('Frame_original', frame)
    # 1
    #cv2.imshow('Frame_menor', resized_frame)
    # 2
    #cv2.imshow('Frame_mascara', back_mask)
    # 3
    #cv2.imshow('Frame_mascara_blur', fgMask)
    # 4
    #cv2.imshow('Frame_mascara_filtros', dilation)

    #--------------------------------------------------
    #cerrar el programa al presionar Esc
    keyboard = cv2.waitKey(30)
    ##################################
    ##################################

    if keyboard == 'q' or keyboard == 27:
        break      