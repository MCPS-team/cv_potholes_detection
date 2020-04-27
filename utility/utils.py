import numpy as np
from skimage import io
import requests
import json
import os

def is_pothole_nearest(a, b):
    # (label, perc, (x,y,width,height))
    a_ybottom = [x[2][1]+(x[2][3]/2) for x in a]
    b_ybottom = [x[2][1]+(x[2][3]/2) for x in b]
    if np.max(a_ybottom) < np.max(b_ybottom):
        return True
        
def save_image(image, path):
    io.imsave(path, image) 
    print("Saving in", path)

# # For MaskR-CNN
# def is_pothole_nearest(rois1, rois2):
#     # rois = [top, left, bottom, right]
#     if np.max(rois1[:,2]) < np.max(rois2[:,2]):
#         return True
#     else:
#         return False


def send_request(url, data):
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    print("Sending data...")
    r = requests.post(url, data=json.dumps(data), headers=headers)
    print("Request accepted wit status code {}".format(r.status_code))


def check_frame_integrity(frames):
    ok_frames = []
    for frame in frames:
        if os.path.isfile(frame):
            ok_frames.append(frame)
        else:
            print ("|WARNING| Image not found at {}".format(frame))
    return ok_frames


def fakes_pothole_obj(frames):
    frames_ids = [{'filename':os.path.basename(frame)} for frame in frames]
    return {"bumpID":-1, 'attached_images':frames_ids}


# def draw_bounding_box(detections):
#     for detection in detections:
#                 label = detection[0]
#                 confidence = detection[1]
#                 pstring = label+": "+str(np.rint(100 * confidence))+"%"