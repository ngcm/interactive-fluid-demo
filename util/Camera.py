from numba import jit
import numpy as np
import cv2

random = np.array(np.power(np.random.rand(16, 8, 3), 3) * 255, dtype=np.uint8)

class Camera:

    def _resize_frame(self, frame, dst, flip=0):
        frame_shape = np.shape(frame)
        frame_crop_height = int(frame_shape[1] / self._ratio)
        crop_offset = (frame_shape[0] - frame_crop_height) // 2
        if crop_offset > 0:
            cropped_frame = frame[crop_offset:-crop_offset, :, :]
        else:
            cropped_frame = frame

        if flip == 1: # horizontal
            cv2.resize(cv2.flip(cropped_frame, 1), self._size, dst=dst)
        elif flip == 2: # verticle
            cv2.resize(cv2.flip(cropped_frame, 0), self._size, dst=dst)
        elif flip == 3: # both
            cv2.resize(cv2.flip(cropped_frame, -1), self._size, dst=dst)
        else:
            cv2.resize(cropped_frame, self._size, dst=dst)

    def __init__(self, size=(640,360), camera_index=0, no_cam_allowed=False):
        self._no_cam_allowed = no_cam_allowed
        self._cap = cv2.VideoCapture(camera_index)
        self._size = size
        self._ratio = size[0] / size[1]
        self._fgbg = cv2.createBackgroundSubtractorKNN()
        self._mask = np.zeros(self._size[::-1], dtype=np.uint8)
        self._input_frame = np.zeros((*self._size[::-1], 3), dtype=np.uint8)
        self._hsv_field = np.zeros((*self._size[::-1], 3), dtype=np.uint8)

        self._last_grey = np.zeros(self._size[::-1], dtype=np.uint8)
        self._current_grey = np.zeros(self._size[::-1], dtype=np.uint8)

        if not self._cap.isOpened():

            # random = np.array(np.power(np.random.rand(16, 8, 3), 3) * 255, dtype=np.uint8)
            self._resize_frame(random, dst=self._input_frame)

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

    @jit
    def update(self, bg_option, mirror_screen, mask_level, mask_width):
        if self._cap.isOpened():
            # update frame if webcam is active
            ret, frame = self._cap.read()
            if ret:
                self._resize_frame(frame, self._input_frame, mirror_screen)
        else:
            # else use a random image
            self._resize_frame(random, self._input_frame, mirror_screen)

        self._last_grey[:] = self._current_grey
        cv2.cvtColor(self._input_frame, cv2.COLOR_BGR2GRAY, dst=self._current_grey)

        if bg_option == 3: # background subtraction
            self._mask[:] = self._fgbg.apply(self._input_frame, learningRate=0.003)
        else:
            self._mask[:] = 0
            cv2.cvtColor(self._input_frame, cv2.COLOR_BGR2HSV, dst=self._hsv_field)
            if bg_option == 2: # hue
                x = np.abs(np.array(self._hsv_field[:,:,0], np.float) / 180 - mask_level)
                self._mask[x > mask_width] = 255
            elif bg_option == 0: # white
                x = np.array(self._hsv_field[:,:,1], np.float) / 255
                x = 1 / mask_width * x * x + mask_level
                y = np.array(self._hsv_field[:,:,2], np.float) / 255
                self._mask[y <= x] = 255
            else: # black
                self._mask[self._hsv_field[:,:,2] > (255 * (1 - mask_level))] = 255

    def reset(self):
        if not self._cap.isOpened():
            random[:] = np.array(np.power(np.random.rand(16, 8, 3), 3) * 255, dtype=np.uint8)

    @property
    def active(self):
        return self._cap.isOpened() or self._no_cam_allowed

    @property
    def shape(self):
        return self._size



    @property
    def input_frame(self):
        return self._input_frame

    @property
    def mask(self):
        return self._mask

    @property
    def current_grey(self):
        return self._current_grey

    @property
    def last_grey(self):
        return self._last_grey

    @jit
    def get_mask(self, size, transpose):
        return cv2.resize(self._mask, size).T
