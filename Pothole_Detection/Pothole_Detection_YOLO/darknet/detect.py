from darknet import performDetect
from glob import glob
import os

files = glob('pothole_weights/data/*.jpg')
OUT_DIR = './pothole_weights/out'

performDetect('', 0.1, "pothole_weights/yolo-pothole-test.cfg", "pothole_weights/yolo-pothole-train_10000.weights", "./cfg/obj.data", initOnly=True)
for file in files:
    print("Read ", file)
    base_name = os.path.basename(file)
    out_path = os.path.join(OUT_DIR, base_name)
    result = performDetect(file, 0.1, "pothole_weights/yolo-pothole-test.cfg", "pothole_weights/yolo-pothole-train_10000.weights", "./cfg/obj.data", saveImage=out_path)
