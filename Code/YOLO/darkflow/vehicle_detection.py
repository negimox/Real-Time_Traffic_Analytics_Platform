import cv2
import numpy as np
from object_detection import ObjectDetection
import math
from darkflow.net.build import  TFNet
import os

options={
   'model':'./cfg/yolo.cfg',        #specifying the path of model
   'load':'./bin/yolov2.weights',   #weights
   'threshold':0.3                  #minimum confidence factor to create a box, greater than 0.3 good
}

tfnet=TFNet(options)
inputPath = os.getcwd() + "/test_images/"
outputPath = os.getcwd() + "/output_images/"

def detectVehicles(filename):
   global tfnet, inputPath, outputPath
   img=cv2.imread(inputPath+filename,cv2.IMREAD_COLOR)
   # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
   result=tfnet.return_predict(img)
   # print(result)
   for vehicle in result:
      label=vehicle['label']   #extracting label
      if(label=="car" or label=="bus" or label=="truck" or label=="bike" or label=="rickshaw"):    # drawing box and writing label
         if(label=="truck"):
            top_left=(vehicle['topleft']['x'],vehicle['topleft']['y'])
            bottom_right=(vehicle['bottomright']['x'],vehicle['bottomright']['y'])
            img=cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)    #green box of width 5
            img=cv2.putText(img,"ambulance",top_left,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)   #image, label, position, font, font scale, colour: black, line width      
         else:
            top_left=(vehicle['topleft']['x'],vehicle['topleft']['y'])
            bottom_right=(vehicle['bottomright']['x'],vehicle['bottomright']['y'])
            img=cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)    #green box of width 5
            img=cv2.putText(img,label,top_left,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)   #image, label, position, font, font scale, colour: black, line width 
   outputFilename = outputPath + "output_" +filename
   cv2.imwrite(outputFilename,img)
   print('Output image stored at:', outputFilename)
   # plt.imshow(img)
   # plt.show()
   # return result

def get_value(desired_value, max_value, default_value=0):
  """
  This function takes a desired value, a maximum value, and an optional default value.
  It returns the minimum of the desired value and the maximum value, ensuring it doesn't exceed the limit.
  """
  return min(desired_value, max_value) if desired_value is not None else default_value


def detectMovingVehicles(filename):
   timer={
      "car":3,
      "truck":5,
      "motorbike":1,
      "bus":7
   }
   od = ObjectDetection()
   class_names=od.classes
   cap = cv2.VideoCapture(inputPath+filename)

   # Initialize count
   count = 0
   center_points_prev_frame = []

   tracking_objects = {}
   track_id = 0

   while True:
      ret, frame = cap.read()
      count += 1
      if not ret:
         break

      # Point current frame
      center_points_cur_frame = []
      vehicle_count=0
      stop_timer=0
      # Detect objects on frame
      (class_ids, scores, boxes) = od.detect(frame)
      for i, box in enumerate(boxes):
         class_id = class_ids[i]
         class_name = class_names[int(class_id)]
         (x, y, w, h) = box
         cx = int((x + x + w) / 2)
         cy = int((y + y + h) / 2)
         center_points_cur_frame.append((cx, cy))
         #print("FRAME N°", count, " ", x, y, w, h)

         if(class_name=="car" or class_name=="truck" or class_name=="bus" or class_name=="bike" or class_name=="truck" or class_name=="rickshaw" or class_name=="motorbike"):    # drawing box and writing label
         # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            if(class_name=="truck"):
               vehicle_count+=1
               stop_timer+=timer[class_name]
               cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
               cv2.putText(frame, "ambulance", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)     
            else:
               vehicle_count+=1
               stop_timer+=timer[class_name]
               cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
               cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)    

            # Display total vehicles detected on top-left
      text = f"Total Vehicles: {vehicle_count}, Signal Timer: {stop_timer}s"
      text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
      box_x = 10
      box_y = text_size[1] + 10
      box_w = text_size[0] + 10
      box_h = text_size[1] + 10
      overlay = frame.copy()
      cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)  # filled black rectangle
      alpha = 0.5  # adjust opacity
      cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
      cv2.putText(frame, text, (box_x, box_y + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


      # Only at the beginning we compare previous and current frame
      if count <= 2:
         for pt in center_points_cur_frame:
               for pt2 in center_points_prev_frame:
                  distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                  if distance < 20:
                     tracking_objects[track_id] = pt
                     track_id += 1
      else:

         tracking_objects_copy = tracking_objects.copy()
         center_points_cur_frame_copy = center_points_cur_frame.copy()

         for object_id, pt2 in tracking_objects_copy.items():
               object_exists = False
               for pt in center_points_cur_frame_copy:
                  distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                  # Update IDs position
                  if distance < 20:
                     tracking_objects[object_id] = pt
                     object_exists = True
                     if pt in center_points_cur_frame:
                           center_points_cur_frame.remove(pt)
                     continue

               # Remove IDs lost
               if not object_exists:
                  tracking_objects.pop(object_id)

         # Add new IDs found
         for pt in center_points_cur_frame:
               tracking_objects[track_id] = pt
               track_id += 1

      # for object_id, pt in tracking_objects.items():
      #    cv2.circle(frame, pt, 5, (0, 0, 255), -1)
      #    cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

      print("Tracking objects")
      print(tracking_objects)


      print("CUR FRAME LEFT PTS")
      print(center_points_cur_frame)


      cv2.imshow("Frame", frame)

      # Make a copy of the points
      center_points_prev_frame = center_points_cur_frame.copy()
      
      # TESTING
      # TESTING
      
      key = cv2.waitKey(1)
      if key == 27:
         break

   cap.release()
   cv2.destroyAllWindows()




for filename in os.listdir(inputPath):
   # if(filename.endswith(".mp4") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
   #    detectVehicles(filename)
   detectMovingVehicles(filename)
   
print("Done!")