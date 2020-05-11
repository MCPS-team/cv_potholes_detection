# cv_potholes_detection
Detect potholes in images using Deep Learning technique (YoloV4 [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet))

# Getting started
Build the docker container with  

```
sudo docker build -f Dockerfile -t docker-cv-pothole-detect .
```

Download and move `yolov4-spp-pothole-train_7000.weights` https://drive.google.com/open?id=1-4BRAxU-ijkp6MlxQ2maetRrnpQ9DjTF into `models/` folder.

## Setting up env
Copy `env` file and rename it in `.env`.
Change variables in `.env` with your custom values.
Ex.
```
# Http port of micro-service
HOST_PORT=5000

# Http url to confirm the completion of job
CALLBACK_URL=http://httpbin.org/post

# Folder in which are stored image's files attached to pothole events
INPUT_IMGS_DIR=/var/www/MCPS/cv_potholes_detection/frames_in

# Images generated by detection with objects bounding box
OUTPUT_IMGS_DIR=/var/www/MCPS/cv_potholes_detection/frames_out

# Host path of model's file directory
MODEL_DIR=/var/www/MCPS/cv_potholes_detection/models

# Model's architecture configuration file
MODEL_CONFIG=models/yolov4-spp-pothole-test.cfg

# Model's pretained weights
MODEL_WEIGHT=models/yolov4-spp-pothole-train_7000.weights

# Discard detected objects with probability less than 
THRESHOLD=0.55
```
## Run
Launch the service using:
```
sh run_docker.sh
```

# Endpoints
```
[POST] /analyze
|Params|: List of pothole events in json format
```





