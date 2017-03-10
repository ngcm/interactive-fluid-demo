import numpy as np
import cv2


class Camera:    
    
    def _resize_frame(self, frame):
        if self._flip:
            return cv2.resize(cv2.flip(frame, 1), self._size)
        else:
            return cv2.resize(frame, self._size)        
    
    def __init__(self, size=(640,480), no_cam_mode=False, flip=True, bg_subtract=False):     
        self._no_cam_mode = no_cam_mode
        self._cap = cv2.VideoCapture(0)
        self._size = size
        self._flip = flip
        self._bg_subtract = bg_subtract
        self._fgbg = cv2.createBackgroundSubtractorKNN()
        self._mask = np.zeros(self._size[::-1], dtype=np.uint8)  
        self._input_frame = np.zeros((*self._size[::-1], 3), dtype=np.uint8) 
        
        if not self._cap.isOpened():
            
            random = np.array(np.power(np.random.rand(16, 8, 3), 3) * 255, dtype=np.uint8)
            self._input_frame = self._resize_frame(random)   
            
            ''' HSV test image
            test_image = np.zeros_like(self._input_frame, dtype=np.uint8)
            x = np.linspace(0, 255, size[0], dtype=np.uint8)
            y = np.linspace(255, 0, size[1], dtype=np.uint8)
            XX, YY = np.meshgrid(x, y)
            test_image[:, :, 1] = XX
            test_image[:, :, 2] = YY
            self._input_frame = cv2.cvtColor(test_image, cv2.COLOR_HSV2BGR)
            '''
            
    def __del__(self):
        self._cap.release()
        
    def update(self, bg_option, mask_level):
        if self._cap.isOpened():
            # update frame if cam is active
            ret, frame = self._cap.read()
            if ret:
                self._input_frame = self._resize_frame(frame)
        elif self._no_cam_mode:
            # else scroll
            None#self._input_frame = self._input_frame.take(range(-1, self._size[1] - 1), axis=0, mode='wrap')
                               
        if self._bg_subtract:
            self._mask = self._fgbg.apply(self._input_frame, learningRate=0.001)
        else:           
            # generate the mask, invert if necessary
            
            self._mask[:] = 0
            hsv = cv2.cvtColor(self._input_frame, cv2.COLOR_BGR2HSV)
            if bg_option == 0:
                x = np.array(hsv[:,:,1], np.float) / 255
                x = 10 * x * x + mask_level
                y = np.array(hsv[:,:,2], np.float) / 255
                self._mask[y <= x] = 255
            else:
                self._mask[hsv[:,:,2] > (255 * mask_level)] = 255
        
    def reset(self):
        if not self._cap.isOpened():           
            random = np.array(np.power(np.random.rand(16, 8, 3), 3) * 255, dtype=np.uint8)
            self._input_frame = cv2.resize(random, self._size)
        
    @property
    def active(self):
        return self._cap.isOpened() or self._no_cam_mode
    
    @property
    def shape(self):
        return self._size
    
    @property
    def input_frame(self):
        return self._input_frame
    
    @property
    def mask(self):
        return self._mask
    
    def get_mask(self, size, transpose):
        return cv2.resize(self._mask, size).T
        
    '''
cv2.startWindowThread()
cv2.namedWindow("FluidSim", flags=cv2.WND_PROP_FULLSCREEN)

a = Camera(no_cam_mode=True)

while True:
    a.update(0, 120)    
    cv2.imshow('FluidSim', a.masked_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
            break


cv2.destroyWindow('FluidSim')
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
'''