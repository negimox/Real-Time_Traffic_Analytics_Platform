import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import time

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt 
import os
import torch

device = torch.device('cpu')

from tqdm import tqdm
from ultralytics import YOLO
#plots
import seaborn as sns
import subprocess
import IPython
from IPython.display import Video, display

model=YOLO('./dnn_model/yolov8n.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('sample4.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()

# cy1=268
# cy2=348

cy1=322
cy2=368

offset=20

vh_down={}
counter=[]


vh_up={}
counter1=[]

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
        


        if cy1<(cy+offset) and cy1 > (cy-offset):
           vh_down[id]=time.time()
        if id in vh_down:
          
           if cy2<(cy+offset) and cy2 > (cy-offset):
             elapsed_time=time.time() - vh_down[id]
             if counter.count(id)==0:
                counter.append(id)
                distance = 10 # meters
                a_speed_ms = distance / elapsed_time
                a_speed_kh = a_speed_ms * 3.6
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                print(f"Speed of vehicle: {a_speed_kh} Km/H")
                
        #####going UP#####     
        if cy2<(cy+offset) and cy2 > (cy-offset):
           vh_up[id]=time.time()
        if id in vh_up:

           if cy1<(cy+offset) and cy1 > (cy-offset):
             elapsed1_time=time.time() - vh_up[id]

 


             if counter1.count(id)==0:
                counter1.append(id)      
                distance1 = 10 # meters
                a_speed_ms1 = distance1 / elapsed1_time
                a_speed_kh1 = a_speed_ms1 * 3.6
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

           

    cv2.line(frame,(274,cy1),(814,cy1),(255,255,255),1)

    cv2.putText(frame,('L1'),(277,320),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)


    cv2.line(frame,(177,cy2),(927,cy2),(255,255,255),1)
 
    cv2.putText(frame,('L2'),(182,367),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    d=(len(counter))
    u=(len(counter1))
    cv2.putText(frame,('goingdown:-')+str(d),(60,90),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    cv2.putText(frame,('goingup:-')+str(u),(60,130),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

# def risize_frame(frame, scale_percent):
#     """Function to resize an image in a percent scale"""
#     width = int(frame.shape[1] * scale_percent / 100)
#     height = int(frame.shape[0] * scale_percent / 100)
#     dim = (width, height)

#     # resize image
#     resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#     return resized

# video = cv2.VideoCapture("./test.mp4")

# ### Configurations
# #Verbose during prediction
# verbose = False
# # Scaling percentage of original frame
# scale_percent = 50


# #-------------------------------------------------------
# # Reading video with cv2
# # video = cv2.VideoCapture(path)

# # Objects to detect Yolo
# class_IDS = [2, 3, 5, 7] 
# # Auxiliary variables
# centers_old = {}
# centers_new = {}
# obj_id = 0 
# veiculos_contador_in = dict.fromkeys(class_IDS, 0)
# veiculos_contador_out = dict.fromkeys(class_IDS, 0)
# end = []
# frames_list = []
# cy_linha = int(1500 * scale_percent/100 )
# cx_sentido = int(2000 * scale_percent/100) 
# offset = int(8 * scale_percent/100 )
# contador_in = 0
# contador_out = 0
# print(f'[INFO] - Verbose during Prediction: {verbose}')


# # Original informations of video
# height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# fps = video.get(cv2.CAP_PROP_FPS)
# print('[INFO] - Original Dim: ', (width, height))

# # Scaling Video for better performance 
# if scale_percent != 100:
#     print('[INFO] - Scaling change may cause errors in pixels lines ')
#     width = int(width * scale_percent / 100)
#     height = int(height * scale_percent / 100)
#     print('[INFO] - Dim Scaled: ', (width, height))
    

# #-------------------------------------------------------
# ### Video output ####
# video_name = 'result.mp4'
# output_path = "rep_" + video_name
# tmp_output_path = "tmp_" + output_path
# VIDEO_CODEC = "MP4V"

# output_video = cv2.VideoWriter(tmp_output_path, 
#                                 cv2.VideoWriter_fourcc(*VIDEO_CODEC), 
#                                 fps, (width, height))


# #-------------------------------------------------------
# # Executing Recognition 
# for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
    
#     # reading frame from video
#     _, frame = video.read()
    
#     #Applying resizing of read frame
#     frame  = risize_frame(frame, scale_percent)
    
#     if verbose:
#         print('Dimension Scaled(frame): ', (frame.shape[1], frame.shape[0]))

#     # Getting predictions
#     y_hat = model.predict(frame, conf = 0.7, classes = class_IDS, device = 0, verbose = False)
    
#     # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
#     boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
#     conf    = y_hat[0].boxes.conf.cpu().numpy()
#     classes = y_hat[0].boxes.cls.cpu().numpy() 
    
#     # Storing the above information in a dataframe
#     positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.boxes, columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
    
#     #Translating the numeric class labels to text
#     labels = [dict_classes[i] for i in classes]
    
#     # Drawing transition line for in\out vehicles counting 
#     cv2.line(frame, (0, cy_linha), (int(4500 * scale_percent/100 ), cy_linha), (255,255,0),8)
    
#     # For each vehicles, draw the bounding-box and counting each one the pass thought the transition line (in\out)
#     for ix, row in enumerate(positions_frame.iterrows()):
#         # Getting the coordinates of each vehicle (row)
#         xmin, ymin, xmax, ymax, confidence, category,  = row[1].astype('int')
        
#         # Calculating the center of the bounding-box
#         center_x, center_y = int(((xmax+xmin))/2), int((ymax+ ymin)/2)
        
#         # drawing center and bounding-box of vehicle in the given frame 
#         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 5) # box
#         cv2.circle(frame, (center_x,center_y), 5,(255,0,0),-1) # center of box
        
#         #Drawing above the bounding-box the name of class recognized.
#         cv2.putText(img=frame, text=labels[ix]+' - '+str(np.round(conf[ix],2)),
#                     org= (xmin,ymin-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=2)
        
#         # Checking if the center of recognized vehicle is in the area given by the transition line + offset and transition line - offset 
#         if (center_y < (cy_linha + offset)) and (center_y > (cy_linha - offset)):
#             if  (center_x >= 0) and (center_x <=cx_sentido):
#                 contador_in +=1
#                 veiculos_contador_in[category] += 1
#             else:
#                 contador_out += 1
#                 veiculos_contador_out[category] += 1
    
#     #updating the counting type of vehicle 
#     contador_in_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_in.items()]
#     contador_out_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_out.items()]
    
#     #drawing the number of vehicles in\out
#     cv2.putText(img=frame, text='N. vehicles In', 
#                 org= (30,30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
#                 fontScale=1, color=(255, 255, 0),thickness=1)
    
#     cv2.putText(img=frame, text='N. vehicles Out', 
#                 org= (int(2800 * scale_percent/100 ),30), 
#                 fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=1)
    
#     #drawing the counting of type of vehicles in the corners of frame 
#     xt = 40
#     for txt in range(len(contador_in_plt)):
#         xt +=30
#         cv2.putText(img=frame, text=contador_in_plt[txt], 
#                     org= (30,xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
#                     fontScale=1, color=(255, 255, 0),thickness=1)
        
#         cv2.putText(img=frame, text=contador_out_plt[txt],
#                     org= (int(2800 * scale_percent/100 ),xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
#                     fontScale=1, color=(255, 255, 0),thickness=1)
    
#     #drawing the number of vehicles in\out
#     cv2.putText(img=frame, text=f'In:{contador_in}', 
#                 org= (int(1820 * scale_percent/100 ),cy_linha+60),
#                 fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=2)
    
#     cv2.putText(img=frame, text=f'Out:{contador_out}', 
#                 org= (int(1800 * scale_percent/100 ),cy_linha-40),
#                 fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=2)

#     if verbose:
#         print(contador_in, contador_out)
#     #Saving frames in a list 
#     frames_list.append(frame)
#     #saving transformed frames in a output video formaat
#     output_video.write(frame)
    
#     #Releasing the video    
#     output_video.release()


#     ####  pos processing
#     # Fixing video output codec to run in the notebook\browser
#     if os.path.exists(output_path):
#         os.remove(output_path)
        
#     subprocess.run(
#         ["ffmpeg",  "-i", tmp_output_path,"-crf","18","-preset","veryfast","-hide_banner","-loglevel","error","-vcodec","libx264",output_path])
#     os.remove(tmp_output_path)