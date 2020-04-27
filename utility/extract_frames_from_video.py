
# Importing all necessary libraries 
import cv2 
import os 
import PIL
from PIL import Image

def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

# Read the video from specified path 
cam = cv2.VideoCapture("./data/vlc-record-2020-03-29-15h38m24s-vlc-record-2020-03-29-15h34m14s-DASH CAM 2016 01 29 .mp4") 
  
try: 
      
    # creating a folder named data 
    if not os.path.exists('data'): 
        os.makedirs('data') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 
  
# frame 
currentframe = 0
jump_frames = 3
while(True): 
      
    # reading from frame 
    ret,frame = cam.read() 
  
    if ret: 
        if currentframe % jump_frames == 0:
            # if video is still left continue creating images 
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name) 

            frame_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(frame_im)
            pil_im.thumbnail((800,800), PIL.Image.ANTIALIAS )
            pil_im = make_square(pil_im, min_size=800)
            # writing the extracted images 
            pil_im.save(name, "JPEG") 
    
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else: 
        break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows()