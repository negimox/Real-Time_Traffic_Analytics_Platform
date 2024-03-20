import cv2
import numpy as np


class ObjectDetection:
    def __init__(self, weights_path="./dnn_model/yolov4.weights", cfg_path="./dnn_model/yolov4.cfg"):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 416

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="./dnn_model/classes.txt"):

        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)

    def get_class_name(self, frame):  # New function to retrieve class name
        (class_ids, scores, boxes) = self.detect(frame)
        if len(class_ids) > 0:  # Check if any objects were detected
            class_id = int(class_ids[0])  # Get the first detected class ID
            return self.classes[class_id]  # Return the corresponding class name
        else:
            return "No object detected"  # Return a default message if no object is found

    def name_detect(self, frame):
        (class_ids, scores, _) = self.detect(frame)
        if len(class_ids) > 0:
            class_id = class_ids[0][0]
            class_name = self.classes[class_id]
            return class_name
        else:
            return "Unknown"

