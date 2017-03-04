# Fluid sim demo prototype

import numpy as np
import cv2

from Camera import Camera
from DensityField import DensityField
from AltSim import Sim
import Options
from FPS_counter import FPS_counter

cv2.startWindowThread()
cv2.namedWindow("window", flags=cv2.WND_PROP_FULLSCREEN)

camera = Camera(no_cam_mode=True)

if(camera.active):
    
    bgOption = Options.Cycle('BG', 'b', ['white', 'black'], 0)
    speedOption = Options.Range('Inflow Speed', ['-','='], [0.02, 1], 0.02, 0.2)
    levelOption = Options.Range('Mask Threshold', ['[',']'], [0, 255], 8, 60)
    smokeStreams = Options.Range('Smoke Streams', [',','.'], [1, 50], 1, 10)
    smokeAmount = Options.Range('Smoke Amount', ['\'','#'], [1, 10], 1, 3)
    debugMode = Options.Cycle('Mode', 'd', ['Normal', 'Debug'], 0)    
    options = [bgOption, speedOption, levelOption, smokeStreams, smokeAmount, debugMode]

    # initialise the fluid sim arrays
    width, height = camera.shape
    
    # sub-sample the webcam image to fit the fluid sim resolution
    step = 2
    sim_shape = (width // step, height // step)
    sim = Sim(sim_shape, 0, 0)
    d = DensityField(sim_shape)
    
    fps = FPS_counter(limit=15)
    
    display_counter = 0 # display values for a short time if they change

    while(True):    
        fps.update()     
        
        if display_counter > 0:
            display_counter -= fps.last_dt
        
        # update input image
        camera.update(bgOption.current, levelOption.current)
         
        # copy the webcam generated mask into the boundary
        box = np.array(cv2.resize(camera.mask, sim_shape).T, dtype=bool)

        # add the bounding box
        box[:, :1] = True
        box[:, -1:] = True
        box[:1, :] = True
        box[-1:, :] = True
        
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
            rgb = np.clip(np.sqrt(d.colour_field.T) * 255, 0, 255)
            sim_render = np.array(cv2.resize(rgb, (width, height)), dtype=np.uint8)
            alpha = cv2.resize(d.alpha.T, (width, height)) / 255 # cv2.cvtColor(sim_render, cv2.COLOR_RGB2GRAY) / (255 * 255)
            output = sim_render * alpha[:,:,np.newaxis] + camera.input_frame * (1/255 - alpha[:,:,np.newaxis])
        else:
            sim.step(fps.last_dt, [])
            rgb = sim.get_velocity_field_as_RGB(0.25).T * 255 # should be 0.5 (i.e. square root), but this shows the lower velocities better
            sim_render = np.array(cv2.resize(rgb, (width, height)), dtype=np.uint8)
            output = cv2.add(cv2.cvtColor(camera.mask, cv2.COLOR_GRAY2BGR), sim_render)
 
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
        
        # use a portion of the remaining frame time to poll key imputs
        key = cv2.waitKey(max(1, int(fps.dt_remaining * 500))) & 0xFF
        
        # update the options (poll for input, cycle)
        for option in options:
            if option.update(key, fps.last_dt):
                display_counter = 1
                         
        # poll for quit, reset
        if key == ord('q'):
            while cv2.waitKey(100) > 0:
                None
            break
        elif key == ord('r'):
            sim.reset()
            d.reset()
            
else:
    print("ERROR: Couldn't capture frame. Is Webcam available/enabled?")
        
# close the window
cv2.destroyAllWindows()
# weirdly, this helps
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)