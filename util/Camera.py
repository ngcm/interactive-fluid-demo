from numba import jit
import numpy as np
import cv2


class Camera:

    def _resize_frame(self, frame):
        frame_shape = np.shape(frame)
        frame_crop_height = int(frame_shape[1] / self._ratio)
        crop_offset = (frame_shape[0] - frame_crop_height) // 2
        if crop_offset > 0:
            cropped_frame = frame[crop_offset:-crop_offset, :, :]
        else:
            cropped_frame = frame

        if self._flip:
            return cv2.resize(cv2.flip(cropped_frame, 1), self._size)
        else:
            return cv2.resize(cropped_frame, self._size)

    def __init__(self, size=(640,360), no_cam_mode=False, flip=True):
        self._no_cam_mode = no_cam_mode
        self._cap = cv2.VideoCapture(0)
        self._size = size
        self._ratio = size[0] / size[1]
        self._flip = flip
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

    @jit
    def update(self, bg_option, mask_level, mask_width):
        if self._cap.isOpened():
            # update frame if cam is active
            ret, frame = self._cap.read()
            if ret:
                self._input_frame = self._resize_frame(frame)
        elif self._no_cam_mode:
            # else scroll
            None#self._input_frame = self._input_frame.take(range(-1, self._size[1] - 1), axis=0, mode='wrap')

        if bg_option == 2:
            # use opencv image background subtraction
            self._mask[:] = self._fgbg.apply(self._input_frame, learningRate=0.003)
        else:
            # generate the mask, invert if necessary
            self._mask[:] = 0
            hsv = cv2.cvtColor(self._input_frame, cv2.COLOR_BGR2HSV)
            if bg_option == 3:
                x = np.abs(np.array(hsv[:,:,0], np.float) / 180 - mask_level)
                self._mask[x > mask_width] = 255
            elif bg_option == 0:
                x = np.array(hsv[:,:,1], np.float) / 255
                x = 1 / mask_width * x * x + mask_level
                y = np.array(hsv[:,:,2], np.float) / 255
                self._mask[y <= x] = 255
            else:
                self._mask[hsv[:,:,2] > (255 * (1 - mask_level))] = 255

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

    @jit
    def get_mask(self, size, transpose):
        return cv2.resize(self._mask, size).T
