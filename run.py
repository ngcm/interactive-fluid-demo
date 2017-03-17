# Fluid sim demo prototype

import numpy as np
import cv2
from numba import jit

from util.Camera import Camera
from util.FPS_counter import FPS_counter
from pynput import keyboard
import util.Options as Options
import sys

if len(sys.argv) > 1 and sys.argv[1] == "C":
    print("using Sim_C")
    from Sim_C import Sim
else:
    print("using Sim_numpy_jit")
    from Sim_numpy_jit import Sim

simResmultiplier = 0.7
fullscreen = True
flip = False

pressed_keys = []
def on_release(key):
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)

cv2.startWindowThread()
cv2.namedWindow("window", flags=cv2.WND_PROP_FULLSCREEN)
if fullscreen:
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

@jit
def run_sim(camera, pressed_keys, simResmultiplier):

    simRes = Options.Range('Sim Res', ['9','0'], [0.1, 2.0], 0.1, simResmultiplier)
    speedOption = Options.Range('Inflow Speed', ['-','='], [0.02, 1], 0.02, 0.2)
    smokeStreams = Options.Range('Smoke Streams', ['[',']'], [1, 50], 1, 27)
    smokeAmount = Options.Range('Smoke Amount', ['\'','#'], [1, 10], 1, 10)
    bgOption = Options.Cycle('BG', 'b', ['white', 'black', 'subtract', 'hue'], 2)
    levelOption = Options.Range('Mask Threshold', ['1','2'], [0, 1], 0.03, 0.4)
    widthOption = Options.Range('Mask Width', ['3','4'], [0, 0.5], 0.01, 0.1)
    mask_render = Options.Cycle('Render Mask', 'm', ['false', 'true'], 1)
    debugMode = Options.Cycle('Mode', 'd', ['Normal', 'Debug'], 0)
    options = [simRes, speedOption, smokeStreams, smokeAmount, bgOption, levelOption, widthOption, mask_render, debugMode]

    # sub-sample the webcam image to fit the fluid sim resolution
    sim = Sim(camera.shape, simRes.current, 0, 0)

    fps = FPS_counter(limit=15)

    display_counter = 0 # display values for a short time if they change

    key_listner = keyboard.Listener(on_press=None, on_release=on_release)
    key_listner.start()

    run = True

    while(run):
        fps.update()

        if simRes.current != simResmultiplier:
            simResmultiplier = simRes.current
            sim = Sim(camera.shape, simRes.current, 0, 0)

        if display_counter > 0:
            display_counter -= fps.last_dt

        # update input image
        camera.update(bgOption.current, levelOption.current, widthOption.current)

        # copy the webcam generated mask into the boundary
        box = np.array(cv2.resize(camera.mask, sim.shape).T, dtype=bool)

        # add the bounding box
        #box[:, :1] = True
        #box[:, -1:] = True
        box[:1, :] = True
        box[-1:, :] = True

        # apply input velocity
        flowwidth = 1 + int(fps.last_dt * speedOption.current / sim._dx[0])
        #sim.set_velocity(np.s_[0, :, :1], speedOption.current)
        #sim.set_velocity(np.s_[0, :, -1:], speedOption.current)
        sim.set_velocity(np.s_[1, :, :flowwidth], speedOption.current)
        sim.set_velocity(np.s_[1, :, -flowwidth:], speedOption.current)
        sim.set_boundary(box)

        # update and render the sim
        sim.udpate(debugMode.current, fps.last_dt, flowwidth,
            smokeStreams.current, smokeAmount.current)
        output = sim.render(debugMode.current, camera, render_mask=(mask_render.current == 1))
        output_shape = np.shape(output)

        # add the GUI
        text_color = (0, 0, 255) if bgOption.current == 0 else (255, 255, 0)
        if debugMode.current == 0 and display_counter <= 0:
            cv2.putText(output, 'd=Debug Mode', (30,output_shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
        else:
            pos = np.array((30,30))
            for option in options:
                cv2.putText(output, str(option), tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
                pos = pos + [0,20]
            cv2.putText(output, str(fps), (output_shape[1] - 100,output_shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
            cv2.putText(output, 'q=Quit, r=Reset', (30,output_shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

        # render the output
        cv2.imshow('window', output)

        for key in pressed_keys:
            # update the options (poll for input, cycle)
            for option in options:
                if option.update(key, fps.last_dt):
                    display_counter = 3

            # poll for quit, reset
            if key == 'q':
                run = False
            elif key == 'r':
                sim.reset()
                camera.reset()

        pressed_keys[:] = []

    key_listner.stop()

camera = Camera(no_cam_mode=True, flip=flip)

if(camera.active):
    run_sim(camera, pressed_keys, simResmultiplier=simResmultiplier)
else:
    print("ERROR: Couldn't capture frame. Is Webcam available/enabled?")

# close the window
cv2.destroyAllWindows()
# weirdly, this helps
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
