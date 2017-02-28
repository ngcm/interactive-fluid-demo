# Fluid sim demo prototype

import numpy as np
import cv2
from numba import jit

from Camera import Camera
from DensityField import DensityField
import StamFluidSim
import AltSim
import MacSim
import Options
from FPS_counter import FPS_counter

cv2.startWindowThread()
cv2.namedWindow("window", flags=cv2.WND_PROP_FULLSCREEN)

camera = Camera(no_cam_mode=True)

if(camera.active):
    
    modeOption = Options.Cycle('Mode', 'm', ['Smoke', 'Velocity'], 1, auto_cycle=True, cycle_time=[20, 10])
    bgOption = Options.Cycle('BG', 'b', ['white', 'black'], 1)
    speedOption = Options.Range('Speed', ['-','='], [0.02, 1], 0.02, 0.2)
    levelOption = Options.Range('Mask Level', ['[',']'], [10, 250], 10, 180)
    smokeAmount = Options.Range('SmokeAmount', ['\'','#'], [1, 10], 1, 3)
    flowMode = Options.Cycle('FlowMode', 'v', ['Wind Tunnel', 'Washing Machine'], 0)
    viscosity = Options.Range('Viscosity', ['9','0'], [0, 1], 0.0001, 0)
    diffusion = Options.Range('Diffusion', ['o','p'], [0, 1], 0.0001, 0)
    
    options = [flowMode, modeOption, bgOption, speedOption, levelOption, smokeAmount]#, viscosity, diffusion]

    # sub-sample the webcam image to fit the fluid sim resolution
    step = 3

    # initialise the fluid sim arrays
    width, height = camera.shape
    sim_shape = (int(width/step), int(height/step))
    diff = 0.001
    visc = 0.0#001
    # sim = StamFluidSim.StamFluidSim(sim_shape, diffusion, viscosity)
    sim = MacSim.Sim(sim_shape, diffusion, viscosity)
    
    d = DensityField(sim_shape)
    
    fps = FPS_counter(15)

    while(True):    
        fps.update()        
        
        mask = np.zeros_like(camera.mask) #camera.get_mask(sim_shape, True)
        if flowMode.current == 1:
            thickness = int(np.sqrt((width//2)**2 + (height//2)**2)) - height//2
            cv2.circle(mask, (width//2, height//2), thickness//2 + height//2, (255), thickness)
        camera.update(mask, bgOption.current, levelOption.current)
         
        # copy the webcam data into the boundary
        box = np.array(cv2.resize(camera.mask, sim_shape).T, dtype=bool)

        # add the bounding box
        box[:, :1] = True
        box[:, -1:] = True
        box[:1, :] = True
        box[-1:, :] = True
                
        flowwidth = 1 + int(fps.last_dt * speedOption.current / sim._dx[0])

        if flowMode.current == 0:  
            sim.set_velocity(np.s_[0, :, :1], speedOption.current)
            sim.set_velocity(np.s_[0, :, -1:], speedOption.current)            
            #box[:flowwidth, :] = True
            #box[-flowwidth:, :] = True
            sim.set_velocity(np.s_[0, :flowwidth, :], speedOption.current)
            sim.set_velocity(np.s_[0, -flowwidth:, :], speedOption.current)
        elif flowMode.current == 1:
            sim.set_velocity(np.s_[:, box], 0)  
            xs, ys = np.meshgrid(np.arange(sim_shape[1]), np.arange(sim_shape[0]))
            r = np.sqrt((xs - sim_shape[1]//2)**2 + (ys - sim_shape[0]//2)**2)
            T = np.logical_or(r < sim_shape[1] // 2 * 0.3, np.logical_and(r > sim_shape[1] // 2 * 0.8, r <= sim_shape[1] // 2)) #np.logical_and(r > sim_shape[1] // 2 * 0.3, np.abs(ys - sim_shape[0]//2) < 10)
            a = np.arctan2(xs[T] - sim_shape[1]//2, ys[T] - sim_shape[0]//2) + np.pi/2
            sim.set_velocity(np.s_[0, T], np.cos(a) * r[T] / 100 * speedOption.current)
            sim.set_velocity(np.s_[1, T], np.sin(a) * r[T] / 100 * speedOption.current)  
        sim.set_boundary(box)              

        if modeOption.current == 0:          
            sim.step(fps.last_dt, d.field)
        elif modeOption.current == 1:
            sim.step(fps.last_dt, [])
        
        d.update(flowMode.current, flowwidth, smokeAmount.current)

        # render
        if modeOption.current == 0:
            rgb = np.clip(np.sqrt(d.field.T) * 255, 0, 255)
        elif modeOption.current == 1:
            rgb = sim.get_velocity_field_as_RGB(0.25).T * 255 # should be 0.5 (i.e. square root), but this shows the lower velocities better
        sim_render = np.array(cv2.resize(rgb, (width, height)), dtype=np.uint8)    

        key = cv2.waitKey(max(1, int(fps.dt_remaining * 500))) & 0xFF
                         
        
        for option in options:
            option.update(key, fps.last_dt)
                         
        if key == ord('q'):
            while cv2.waitKey(100) > 0:
                None
            break
        elif key == ord('r'):
            sim.reset()
            d.reset()
             
        if bgOption.current == 0:
            output = cv2.subtract(camera.masked_frame, sim_render)
        else:
            output = cv2.add(camera.masked_frame, sim_render)
        
        text_color = (0, 0, 0) if bgOption.current == 0 else (255, 255, 255)
        
        pos = np.array((30,30))
        for option in options:
            cv2.putText(output, str(option), tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, 0.25, text_color)
            pos = pos + [0,20]
            
        cv2.putText(output, str(fps), (510,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)            
        cv2.putText(output, 'q=quit, r=reset', (30,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
        textSize, _ = cv2.getTextSize(modeOption.current_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
        cv2.putText(output, modeOption.current_name, ((width - textSize[0])//2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.imshow('window', output)
            
else:
    print("ERROR: Couldn't capture frame. Is Webcam available/enabled?")
        
# release the capture decive
# cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)