from time import time
from .darknet import darknet
import os

class PotholeDetection():
    def __init__(self,  configPath="./cfg/yolov3.cfg", weightPath="yolov3.weights", metaPath='cfg/obj.data', saveDir='./'):
        super().__init__()
        self.configPath = configPath
        self.weightPath = weightPath
        self.metaPath = metaPath
        self.saveDir = saveDir

        darknet.performDetect('', configPath=self.configPath,
                              weightPath=self.weightPath, metaPath=self.metaPath, initOnly=True)

    def detect(self, image, thresh=0.1, save=True):
        if save:
            saveImage = os.path.join(self.saveDir, os.path.basename(image))
        else:
            saveImage = None
            
        detections = darknet.performDetect(image, thresh=thresh, configPath=self.configPath,
                                           weightPath=self.weightPath, metaPath=self.metaPath, showImage=True, saveImage=saveImage)

        # detections['detections'] = [(label, perc, (x,y, width, height))]
        return detections


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Test model PorholeDetection')
    parser.add_argument('config_path',  type=str,
                        help='path of darknet config', required=True)
    parser.add_argument('weights_path',  type=str,
                        help='path of darknet weights', required=True)
    parser.add_argument('meta_path',  type=str,
                        help='path of meta data', required=True)
    parser.add_argument('image_path',  type=str,
                        help='path of image to analyze', required=True)
    parser.add_argument('--show-image', action='store_true',
                        help="show image")
    parser.add_argument('--image-only', action='store_true',
                        help="make image only")
    args = parser.parse_args()

    start = time()
    print("Initializing Model...")
    model = PotholeDetection(configPath=args.confing_path, weightPath=args.weights_path,
                             metaPath=args.meta_path, showImage=args.show_image, makeImageOnly=args.makeImageOnly)
    print("Time elapsed: {}s".format(time()-start))

    start = time()
    print("Analyzing image...")
    model.detect(args.image_path, thresh=0.6)
    print("Time elapsed: {}s".format(time()-start))

