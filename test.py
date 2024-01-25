from api import Face_Encode_Model
import cv2
import matplotlib.pyplot as plt
import os
import time

face_model = Face_Encode_Model('/home/matheus_levy/workspace/Face_Recognition/weights/vgg_embedding.keras')

face_model.fit_dataset('/home/matheus_levy/workspace/Face_Recognition/verify_dataset')

img_angela= cv2.imread('/home/matheus_levy/workspace/Face_Recognition/input_dataset/angela2.jpeg')
img_darlene= cv2.imread('/home/matheus_levy/workspace/Face_Recognition/input_dataset/darlene3.jpg')
img_tyrel= cv2.imread('/home/matheus_levy/workspace/Face_Recognition/input_dataset/tyrell2.jpg')
img_ellie= cv2.imread('/home/matheus_levy/workspace/Face_Recognition/input_dataset/ellie2.jpg')

img_angela= cv2.cvtColor(img_angela, cv2.COLOR_BGR2RGB)
img_darlene= cv2.cvtColor(img_darlene, cv2.COLOR_BGR2RGB)
img_tyrel= cv2.cvtColor(img_tyrel, cv2.COLOR_BGR2RGB)
img_ellie= cv2.cvtColor(img_ellie, cv2.COLOR_BGR2RGB)

tempo_inicio = time.time()
results_angela = face_model.match_image(img_angela, True, 1)
results_darlene = face_model.match_image(img_darlene, True, 1)
results_tyrel = face_model.match_image(img_tyrel, True, 1)
results_ellie = face_model.match_image(img_ellie, True, 1)
tempo_fim = time.time()

print(f'Match Angela: {results_angela}')
print(f'Match Darlene: {results_darlene}')
print(f'Match Tyrel: {results_tyrel}')
print(f'Match Ellie: {results_ellie}')
print(f'Time: {tempo_fim - tempo_inicio}s')