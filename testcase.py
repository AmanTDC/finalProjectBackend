import cv2
import numpy as np
from flask import jsonify
import requests
import codecs,json
folder = "10021"
imgs = np.array(np.transpose(cv2.imread(folder+"//00001.jpg"),(2,0,1))).reshape(3,1,100,176)
    
for i in range(2,38,3):
    if(i<=9):
        toappend = "0"+str(i)
    else:
        toappend = str(i)
        

    
    imgs = np.column_stack((imgs,(np.transpose(cv2.imread(folder+"//000"+toappend+".jpg"),(2,0,1))).reshape(3,1,100,176)))
    imgs = np.array(imgs)
imgs2 = imgs.reshape(1,1,3,13,100,176)

file_path = "path.json"
to_send = imgs2.tolist() ### this saves the array in .json format
    #to_send = jsonify(imgs2.to_list())
#print("Error in jsonifying")
url = "http://localhost:5000/post"
print("Processing end")
print(requests.post(url,json={'imgs':to_send}).text)