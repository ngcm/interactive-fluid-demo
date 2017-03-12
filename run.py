# Fluid sim demo prototype

import numpy as np
import cv2
from numba import jit

from util.Camera import Camera
from util.FPS_counter import FPS_counter
from DensityField import DensityField
from pynput import keyboard
import util.Options as Options
import sys

if len(sys.argv) > 1 and sys.argv[1] == "C":
    print("using Sim_C")
    from Sim_C import Sim
else:
    print("using Sim_numpy_jit")
    from Sim_numpy_jit import Sim
    
pressed_keys = []
def on_release(key):
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)

cv2.startWindowThread()
cv2.namedWindow("window", flags=cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

@jit
def run_sim(camera, pressed_keys, simResmultiplier):     
    
    bgOption = Options.Cycle('BG', 'b', ['white', 'black'], 1)
    speedOption = Options.Range('Inflow Speed', ['-','='], [0.02, 1], 0.02, 0.2)
    levelOption = Options.Range('Mask Threshold', ['[',']'], [0, 1], 0.1, 0.4)
    smokeStreams = Options.Range('Smoke Streams', [',','.'], [1, 50], 1, 16)
    smokeAmount = Options.Range('Smoke Amount', ['\'','#'], [1, 10], 1, 10)
    simRes = Options.Range('Sim Res', ['9','0'], [0.1, 2.0], 0.1, simResmultiplier)
    debugMode = Options.Cycle('Mode', 'd', ['Normal', 'Debug'], 0)    
    options = [bgOption, speedOption, levelOption, smokeStreams, smokeAmount, simRes, debugMode]

    # initialise the fluid sim arrays
    width, height = camera.shape
    
    # sub-sample the webcam image to fit the fluid sim resolution
    sim_shape = (int(width * simRes.current), int(height * simRes.current))
    sim = Sim(sim_shape, 0, 0)
    d = DensityField(sim_shape)
    
    fps = FPS_counter(limit=15)
    
    display_counter = 0 # display values for a short time if they change

    key_listner = keyboard.Listener(on_press=None, on_release=on_release)
    key_listner.start()
        
    run = True
    
    while(run):
        fps.update()     
        
        if simRes.current != simResmultiplier:
            simResmultiplier = simRes.current
            sim_shape = (int(width * simRes.current), int(height * simRes.current))
            sim = Sim(sim_shape, 0, 0)
            d = DensityField(sim_shape)
        
        if display_counter > 0:
            display_counter -= fps.last_dt
        
        # update input image
        camera.update(bgOption.current, levelOption.current)
         
        # copy the webcam generated mask into the boundary
        box = np.array(cv2.resize(camera.mask, sim_shape).T, dtype=bool)

        # add the bounding box
        #box[:, :1] = True
        #box[:, -1:] = True
        #box[:1, :] = True
        #box[-1:, :] = True
        
        # apply input velocity
        flowwidth = 1 + int(fps.last_dt * speedOption.current / sim._dx[0])
        #sim.set_velocity(np.s_[0, :, :1], speedOption.current)
        #sim.set_velocity(np.s_[0, :, -1:], speedOption.current)  
        sim.set_velocity(np.s_[0, :flowwidth, :], speedOption.current)
        sim.set_velocity(np.s_[0, -flowwidth:, :], speedOption.current)
        sim.set_boundary(box)              
             
        # update and render the sim
        if debugMode.current == 0:
            d.update(flowwidth, smokeStreams.current, smokeAmount.current)
            sim.step(fps.last_dt, d.field)
            rgb = np.clip(np.sqrt(d.get_colour_field().T) * 255, 0, 255)
            sim_render = np.array(cv2.resize(rgb, (width, height)), dtype=np.uint8)
            alpha = cv2.resize(d.get_alpha().T, (width, height)) / 255 # cv2.cvtColor(sim_render, cv2.COLOR_RGB2GRAY) / (255 * 255)
            output = sim_render * alpha[:,:,np.newaxis] + camera.input_frame * (1/255 - alpha[:,:,np.newaxis])
            # output[:height//4,:width//4] = cv2.resize(camera.mask, (width//4,height//4))[:,:,np.newaxis]
        else:
            sim.step(fps.last_dt, [])
            velrgb = cv2.cvtColor(sim.get_velocity_field_as_HSV(0.25).T, cv2.COLOR_HSV2BGR) # should be 0.5 (i.e. square root), but this shows the lower velocities better
            vel_render = np.array(cv2.resize(velrgb, (width, height)), dtype=np.uint8)
            pressrgb = sim.get_pressure_as_rgb().T * 512
            press_render = np.array(cv2.resize(pressrgb, (width, height)), dtype=np.uint8)
            press_render = cv2.blur(press_render, (20, 20), press_render)
            output = cv2.add(cv2.cvtColor(camera.mask // 8, cv2.COLOR_GRAY2BGR), vel_render)
            output = cv2.add(output, press_render, output, mask=camera.mask)
 
        # add the GUI       
        text_color = (0, 0, 255) if bgOption.current == 0 else (255, 255, 0)
        if debugMode.current == 0 and display_counter <= 0:   
            cv2.putText(output, 'd=Debug Mode', (30,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
        else:
            pos = np.array((30,30))
            for option in options:
                cv2.putText(output, str(option), tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
                pos = pos + [0,20]
            cv2.putText(output, str(fps), (520,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)            
            cv2.putText(output, 'q=Quit, r=Reset', (30,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
            
        # render the output
        cv2.imshow('window', output)

        for key in pressed_keys:
            # update the options (poll for input, cycle)
            for option in options:
                if option.update(key, fps.last_dt):
                    display_counter = 1
                             
            # poll for quit, reset
            if key == 'q':
                run = False
            elif key == 'r':
                sim.reset()
                d.reset()
                camera.reset()
                
        pressed_keys[:] = []
            
                
    key_listner.stop()

camera = Camera(no_cam_mode=True)

if(camera.active):
    run_sim(camera, pressed_keys, simResmultiplier=1.0)
else:
    print("ERROR: Couldn't capture frame. Is Webcam available/enabled?")
        
# close the window
cv2.destroyAllWindows()
# weirdly, this helps
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)