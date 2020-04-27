import glob
import cv2
import os
import numpy as np
from time import time

orb = cv2.ORB_create()
bf = cv2.BFMatcher()


def sort_frame_names(filename):
    id = os.path.basename(filename)
    id = id[5:-4]
    return int(id)


def extract_key_images(frames, lambda_match=300):
    descriptors = []
    take_frames = []

    if len(frames) < 0:
        return []

    if len(frames) < 5:
        for frame in frames:
            take_frames.append(frame)
        return take_frames

    for frame in frames:
        img = cv2.imread(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        if len(descriptors) > 0:
            matches = bf.knnMatch(descriptors[-1], des, k=2)

            is_similar = 0
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    is_similar += 1
                    if is_similar > lambda_match:
                        break
            if is_similar < lambda_match:
                descriptors.append(des)
                take_frames.append(frame)
        else:
            descriptors.append(des)
            take_frames.append(frame)

    return take_frames


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    # from Pothole_Detection import PotholeDetection
    from config import config
    # model = PotholeDetection(configPath=config.config, weightPath=config.weight, metaPath=config.meta, saveDir=config.out_dir)
    print(os.getcwd())
    frames = list(glob.glob('./frames_in/*.jpg'))
    print(os.path.join('./frames_in/', "*.jpg"))
    print("Total frames {}".format(len(frames)))
    # frames.sort(key = sort_frame_names, reverse = False)
    frames = extract_key_images(frames, lambda_match=250)
    print("Frames to be analyzed, ", len(frames))
    # optim_detections = []
    # for frame in frames[10:]:
    #     start = time()
    #     detections = model.detect(frame, save=True)
    #     print("Running detection in {}s".format(time()-start))

    #     print(detections['detections'], detections['caption'])

    #     if len(detections)>0:
    #         if len(optim_detections)>0:
    #             if is_pothole_lower(optim_detections, detections):
    #                 optim_detections = detections
    #             else:
    #                 save_image(optim_detections['image'])
    #                 optim_detections = detections
    #         else:
    #             optim_detections = detections


# #################
# # For Mask-RCNN
# #################
# if __name__ == '__main__':
#     import sys
#     sys.path.append('../')
#     from PotholeDetection import PotholeDetection
#     from config import config
#     model = PotholeDetection(configPath=config.config, weightPath=config.weight, saveDir=config.out_dir)

#     frames = glob.glob(os.path.join('../data',"*.jpg"))
#     frames.sort(key = sort_frame_names, reverse = False)
#     frames = extract_key_images(frames)
#     print("Frames to be analyzed, ", len(frames))
#     take_frame_out = None
#     for frame in frames[10:]:
#         image = model.read_image(frame)
#         find, r = model.detect(image, save=False)
#         # model.save_image(image, r['masks'], os.path.basename(frame))
#         if find:
#             if take_frame_out is not None:
#                 if is_pothole_lower(take_frame_out['r']['rois'], r['rois']):
#                     take_frame_out =  {"r":r, "image":image, "frame_id":frame}
#                 else:
#                     model.save_image(take_frame_out['image'], take_frame_out['r']['masks'], os.path.basename(take_frame_out['frame_id']))
#                     take_frame_out =  {"r":r, "image":image, "frame_id":frame}
#             else:
#                 take_frame_out = {"r":r, "image":image, "frame_id":frame}
