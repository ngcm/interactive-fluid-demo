# Fluid sim demo prototype

import numpy as np
import cv2
from numba import jit

from util.Camera import Camera
from util.FPS_counter import FPS_counter
from pynput import keyboard
import util.Options as Options

# from configuration import Sim
import configuration

# Using a queue to decouple key press from update action
pressed_keys = []
def on_release(key):
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)

cv2.startWindowThread()
cv2.namedWindow("window", flags=cv2.WND_PROP_FULLSCREEN)

camera = Camera(camera_index=configuration.camera_index, no_cam_allowed=True)

# initialise run-time configurable Options
# (display name, keys, range, step, initial value)
fullscreen = Options.Cycle('Fullscreen', 'f', ['Window', 'Fullscreen'], configuration.fullscreen)
mirror_screen = Options.Cycle('Mirror Screen', 'g', ['Normal', 'Horizontal', 'Verticle', 'Both'], configuration.mirror_screen)
render_mask = Options.Cycle('Render Mask', 'm', ['false', 'true'], configuration.render_mask)

bg_mode = Options.Cycle('BG', 'b', ['white', 'black', 'hue', 'bg subtract'], configuration.bg_mode)
mask_level = Options.Range('Mask Threshold', ['1','2'], [0, 1], 0.03, configuration.mask_level)
mask_width = Options.Range('Mask Width', ['3','4'], [0, 0.5], 0.01, configuration.mask_width)

sim_res_multiplier = Options.Range('Sim Res', ['9','0'], [0.1, 2.0], 0.1, configuration.sim_res_multiplier)
flow_speed = Options.Range('Flow Speed', ['-','='], [0.02, 1], 0.02, configuration.flow_speed)
flow_direction = Options.Cycle('Flow Direction', 'p', ['right', 'down', 'left', 'up'], configuration.flow_direction)
num_smoke_streams = Options.Range('Smoke Streams', ['[',']'], [1, 50], 1, configuration.num_smoke_streams)
smoke_percentage = Options.Range('Smoke Amount', ['\'','#'], [0.1, 1], 0.1, configuration.smoke_percentage)

debugMode = Options.Cycle('Mode', 'd', ['Normal', 'Debug'], 0)

# add to a list to update and display
options = [fullscreen, mirror_screen, render_mask,
    bg_mode, mask_level, mask_width,
    sim_res_multiplier, flow_speed, flow_direction, num_smoke_streams, smoke_percentage,
    debugMode]

def run_sim():
    fps = FPS_counter(limit=30)

    display_counter = 0 # display values for a short time if they change

    key_listner = keyboard.Listener(on_press=None, on_release=on_release)
    key_listner.start()

    run = True

    while(run):
        fps.update()

        if display_counter > 0:
            display_counter -= fps.last_dt

        # Always true on first iteration. Sub-sample the webcam image to fit the
        # fluid sim resolution. Update when option changes.
        if sim_res_multiplier.get_has_changed(reset_change_flag=True):
            sim = configuration.Sim(camera.shape, sim_res_multiplier.current, 0, 0)

        # Always True on first iteration. Update fullscreen if option changed
        if fullscreen.get_has_changed(reset_change_flag=True):
            if fullscreen.current:
                cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, 1)
            else:
                cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, 0)

        if debugMode.get_has_changed(reset_change_flag=True):
            sim.mode = debugMode.current

        # if flow direction changes, reset, else things get messy
        if flow_direction.get_has_changed(reset_change_flag=True):
            sim.reset()
            camera.reset()

        # update input image
        camera.update(bg_mode.current, mirror_screen.current,
            mask_level.current, mask_width.current)

        sim.set_velocity(fps.last_dt, flow_speed.current,
            flow_direction.current, num_smoke_streams.current,
            smoke_percentage.current)

        # copy the webcam generated mask into the boundary
        boundary = np.array(cv2.resize(camera.mask, sim.shape).T, dtype=bool)
        sim.set_boundary(boundary, flow_direction.current)

        # update and render the sim
        sim.udpate(fps.last_dt)
        output = sim.render(camera, render_mask=(render_mask.current == 1))
        output_shape = np.shape(output)

        # add the GUI
        text_color = (0, 0, 255) if bg_mode.current == 0 else (255, 255, 0)
        if debugMode.current == 0 and display_counter <= 0:
            cv2.putText(output, 'd=Debug Mode', (30,output_shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color)
        else:
            pos = np.array((30,30))
            for option in options:
                cv2.putText(output, str(option), tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color)
                pos = pos + [0,20]
            cv2.putText(output, str(fps), (output_shape[1] - 80,output_shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color)
            cv2.putText(output, 'q=Quit, r=Reset', (30,output_shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color)

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

if(camera.active):
    run_sim()
else:
    print("ERROR: Couldn't capture frame. Is Webcam available/enabled?")

# close the window
cv2.destroyAllWindows()
# weirdly, this helps
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
