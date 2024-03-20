# Real-Time Traffic Analytics

Made for "Innovate or Evaporate" 24 hour hackathon.

Our proposed system addresses the challenge of real-time traffic density calculation by leveraging image processing and object detection. It utilizes images captured from CCTV cameras at traffic junctions as input data. The system comprises three core modules:

### 1. Vehicle Detection Module:

This module is responsible for accurately identifying and counting vehicles within the camera footage.

### 2. Signal Switching Algorithm:

Based on the real-time traffic density data provided by the Vehicle Detection Module, this module dynamically optimizes traffic light timing to improve traffic flow.

### 3. Speed Estimation Algorithm

Two reference lines are drawn at a known distance apart on the image. The system tracks when a vehicle crosses each line and measures the time difference. Finally, by dividing this time by the known distance between the lines, the vehicle's speed is estimated. This approach leverages YOLO's efficient vehicle detection but relies on accurate camera calibration and consistent detection for reliable speed estimates.

### 3. Simulation Module (Optional):

This optional module can be used to simulate traffic scenarios based on historical data and the real-time vehicle detection results. This allows for testing and refining the Signal Switching Algorithm before real-world deployment.

 ## Vehicle Detection Preview
https://github.com/negimox/Real-Time_Traffic_Analytics_Platform/assets/54482209/7941c43b-ca95-4308-8ac1-0ae5deba3821



 ## Speed Estimation Preview
https://github.com/negimox/Real-Time_Traffic_Analytics_Platform/assets/54482209/01d5bd72-e864-4ce5-9019-d4e0c71ec252

