import numpy as np
import cv2

class Camera:
    
    def __init__(self, size=(640,480), no_cam_mode=False):     
        self._no_cam_mode = no_cam_mode
        self._cap = cv2.VideoCapture(0)
        self._size = size
        
        self._input_frame = np.zeros((*self._size[::-1], 3), dtype=np.uint8)        
        if not self._cap.isOpened():           
            random = np.array(np.random.rand(24, 32, 3) * 255, dtype=np.uint8)
            self._input_frame = cv2.resize(random, self._size)
            
        self._mask = np.zeros(self._size[::-1], dtype=np.uint8)
        self._masked_frame = np.zeros_like(self._input_frame)
        
    def __del__(self):
        self._cap.release()
        
    def update(self, mask, bg_option, mask_level):
        if self._cap.isOpened():
            # update frame if cam is active
            ret, frame = self._cap.read()
            if ret:
                self._input_frame = cv2.resize(cv2.flip(frame, 1), self._size)
        elif self._no_cam_mode:
            # else scroll
            self._input_frame = self._input_frame.take(range(-1, self._size[1] - 1), axis=0, mode='wrap')
                               
        # generate the mask, invert if necessary
        self._mask = cv2.cvtColor(self._input_frame, cv2.COLOR_RGB2GRAY)
        
        if bg_option == 0:
            self._mask = cv2.subtract(self._mask, mask)
        else:
            self._mask = cv2.add(self._mask, mask)
        
        _, self._mask = cv2.threshold(self._mask, mask_level, 255, 
                                   cv2.THRESH_BINARY if bg_option == 1 else 
                                   cv2.THRESH_BINARY_INV)
                
        self._masked_frame[:] = 0 if bg_option == 1 else 255
        self._masked_frame = cv2.add(self._input_frame, 0, dst=self._masked_frame, mask=self._mask)
        
    @property
    def active(self):
        return self._cap.isOpened() or self._no_cam_mode
    
    @property
    def shape(self):
        return self._size
        
    @property
    def masked_frame(self):
        return self._masked_frame
    
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