import glob
import os
import numpy as np
from time import time
from copy import deepcopy
from Pothole_Detection import PotholeDetection
from utility import save_image, is_pothole_nearest, extract_key_images, sort_frame_names, check_frame_integrity
from config import config
from tqdm import tqdm
import pandas as pd

model = PotholeDetection(configPath=config.model_config, weightPath=config.model_weight, metaPath=config.meta, saveDir=config.out_dir)

def inference_on_events(pothole_events, clean=False, remove_useless=True, verbose=1):
    ''' Filter out events in which aren't found potholes.
    Each selected image is unique, 
    if two events share the same images with potholes only the last event will be taken.
    Analyze events in LIFO way.
     '''
    # List of yet analyzed images
    analyzed_images = []
    # Frame that can be assigned to events
    useful_frames = []
    # Frame that be returned
    taken_events = []
    # Images saved to file during analysis
    saved_images = []
    verbose_time = []
    stats = {"key_images":0, "lower_prob":0, "found_target":0, "not_found_target":0, "not_found_path":0, "found_path":0}
    # Analize events in reversed order (from last to first)
    for event in tqdm(list(reversed(pothole_events))):
        # Join image name to path location
        if verbose:
            print("# EVENT", event.bumpID)
        if len(event.attached_images)<0:
            if verbose: print("No attached frames, skip to next event.")
            continue
        image_paths = [os.path.join(config.in_dir, image['filename']) for image in event.attached_images]
        # Filter out yet analyzed images
        not_analyzed = [x for x in image_paths if x not in analyzed_images]
        # Check integrity of images
        ok_images_path = check_frame_integrity(not_analyzed)
        # Remove similar-redoundant images
        # Reverse and then revereverse, so we can matain from the nearest image
        key_images_path = list(reversed(extract_key_images(list(reversed(ok_images_path)), lambda_match=0.7)))
        stats['key_images'] += len(key_images_path)
        stats['not_found_path'] += len(not_analyzed)-len(ok_images_path)
        stats['found_path'] += len(ok_images_path)
        
        print("-- Extracted key images not yet analyzed:", len(key_images_path))
        # Detect pothole in key images
        for image_path in key_images_path:
            if verbose:
                print("-> Image: {}".format(image_path))
                start = time()
            out = model.detect(image_path, save=False)
            # out['detections'] is [(label, prob, (x,y, width, height)), (...), ...]
            # get higher probability found
            prob = max([x[1] for x in out['detections']]) if len(out['detections'])>0 else 0
            if prob > config.threshold:
                filename = os.path.basename(image_path)
                save_image(out['image'], os.path.join(config.out_dir, filename))
                useful_frames.append({'probability':prob, 'filename':filename})
                saved_images.append(filename)
                stats['found_target'] += 1
            else:
                if prob==0:
                    stats['not_found_target'] += 1
                else:
                    stats['lower_prob'] += 1
            if verbose:
                verbose_time.append(time()-start)
        if verbose: print("|i| Analyzed {} sec/image".format(np.mean(verbose_time)))

        # Add to yet analyzed all images in event
        analyzed_images += not_analyzed
        
        # Find the frame with higher probability in the list of event image
        find_best_frame = False
        best_frame = None
        while not find_best_frame and len(useful_frames)>0:
            best_frame_index = np.argmax([x['probability'] for x in useful_frames])
            best_frame = useful_frames[best_frame_index]
            if any(best_frame['filename']==img['filename'] for img in event.attached_images):
                find_best_frame = True
            del useful_frames[best_frame_index]
        
        # If find best frame override the others, 
        # else empty the list
        if find_best_frame:
            event.attached_images = [best_frame]
            taken_events.append(event)
        else:
            print("|WARNING| Pothole not detected for event")

    # Remove useless images from disk, that was not assigned to any events
    if remove_useless:
        new_events_images = list(np.array([[e['filename'] for e in event.attached_images] for event in taken_events]).flat)
        for filename in saved_images:
            file_path = os.path.join(config.out_dir, filename)
            if filename not in new_events_images and os.path.isfile(file_path):
                os.remove(file_path)  
                if (verbose):
                    print("Useless image {} deleted.".format(file_path))
        
    # Remove all analyzed original images
    if clean:
        for image_path in analyzed_images:
            if os.path.isfile(image_path):
                os.remove(image_path)
                if (verbose):
                    print("Image {} deleted.".format(image_path))

    print("|Stats|")
    print(stats)
    print("Unique images in events {}".format(len(analyzed_images)))
    print("Input events: {} - Output events {}".format(len(pothole_events), len(taken_events)))
    # Restore list to original order
    return list(reversed(taken_events))
                
                
if __name__ == '__main__':
    frames = glob.glob(os.path.join('./test/data',"*.jpg"))
    frames.sort(key = sort_frame_names, reverse = False) 
    frames = extract_key_images(frames)
    print("Frames to be analyzed, ", len(frames))
    optim_detections = {'detections':[], 'image':None, 'caption':None}
    for frame in frames[10:]:
        start = time()
        detections = model.detect(frame, save=False)
        print("Running detection in {}s".format(time()-start))

        print(detections['detections'], detections['caption'])
        
        if len(detections['detections'])>0:
            if len(optim_detections['detections'])>0:
                if is_pothole_nearest(optim_detections['detections'], detections['detections']):
                    optim_detections = detections
                else:
                    save_image(optim_detections['image'], os.path.join(config.out_dir, os.path.basename(frame)))
                    optim_detections = detections
            else:
                optim_detections = detections