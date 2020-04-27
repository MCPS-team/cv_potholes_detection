from darknet import performDetect
from glob import glob
import os

cfg = "pothole_weights/pothole-yolov3-tiny.cfg"
weight = "pothole_weights/pothole-yolov3-tiny.weights"
meta = "./cfg/obj.data"
result = performDetect('', 0.3, cfg, weight, meta, initOnly=True)

files = glob('pothole_weights/data/*.jpg')
OUT_DIR = './pothole_weights/out'
for file in files:
    base_name = os.path.basename(file)
    out_path = os.path.join(OUT_DIR, base_name)
    result = performDetect(file, 0.3, cfg, weight, meta, saveImage=out_path)
    print("Saved in {}, result= {}".format(out_path, result))
