# Fluid sim demo prototype

import time
import numpy as np
import cv2

import Camera
import StamFluidSim
import AltSim
import util
import Options

cv2.startWindowThread()
cv2.namedWindow("window", flags=cv2.WND_PROP_FULLSCREEN)

camera = Camera.Camera(no_cam_mode=True)

ltime = time.time()
if(camera.active):
    
    modeOption = Options.CycleOption('Mode', 'm', ['smoke', 'velocity'], 1)
    bgOption = Options.CycleOption('BG', 'b', ['white', 'black'], 1)
    speedOption = Options.RangeOption('Speed', ['-','='], [0.05, 3], 0.05, 0.4)
    levelOption = Options.RangeOption('Mask Level', ['[',']'], [10, 250], 10, 180)
    smokeAmount = Options.RangeOption('SmokeAmount', ['\'','#'], [1, 10], 1, 2)
    flowMode = Options.CycleOption('FlowMode', 'v', ['Wind Tunnel', 'Washing Machine'], 0)
    viscosity = Options.RangeOption('Viscosity', ['9','0'], [0, 1], 0.0001, 0)
    diffusion = Options.RangeOption('Diffusion', ['o','p'], [0, 1], 0.0001, 0)
    
    options = [flowMode, modeOption, bgOption, speedOption, levelOption, smokeAmount, viscosity, diffusion]

    # sub-sample the webcam image to fit the fluid sim resolution
    step = 4

    # initialise the fluid sim arrays
    width, height = camera.shape
    sim_shape = (int(width/2), int(height/2))
    diff = 0.001
    visc = 0.0#001
    # sim = StamFluidSim.StamFluidSim(sim_shape, diffusion, viscosity)
    sim = AltSim.Sim(sim_shape, diffusion, viscosity)
    
    d = np.zeros((3, *sim_shape), dtype=np.float32)
    
    ltime = time.time()

    while(True):    
        mask = np.zeros_like(camera.mask) #camera.get_mask(sim_shape, True)
        if flowMode.current == 1:
            thickness = int(np.sqrt((width//2)**2 + (height//2)**2)) - height//2
            cv2.circle(mask, (width//2, height//2), thickness//2 + height//2, (255), thickness)
        camera.update(mask, bgOption.current, levelOption.current)
        
        curtime = time.time()
        dt = curtime - ltime
        fps = 1 / (curtime - ltime)    
        ltime = curtime
            
        # copy the webcam data into the boundary
        
        box = np.array(cv2.resize(camera.mask, sim_shape).T, dtype=bool)

        # add the bounding box
        box[:, :1] = True
        box[:, -1:] = True
        box[:1, :] = True
        box[-1:, :] = True

        flow = int(speedOption.current * dt / sim._dx[0])
        if flowMode.current == 0:            
            r = range(10, sim_shape[1], 10)
            x = np.s_[1:2 + flow]
        elif flowMode.current == 1:            
            r = range(10, sim_shape[1] // 2, 10)
            x = sim_shape[0] //2
            x = np.s_[x - flow // 2:x + 1 + flow // 2]
        for i in r:
            s = np.s_[i:i + smokeAmount.current]
            amount = (np.random.rand() + 1) * smokeAmount.current * dt
            
            d[0, x, s] = np.min([d[0, x, s] + amount, np.ones_like(d[0, x, s], dtype=np.float32)], axis=0) * (2 if i % 3 == 0 else 1)
            d[1, x, s] = np.min([d[1, x, s] + amount, np.ones_like(d[1, x, s], dtype=np.float32)], axis=0) * (2 if i % 3 == 1 else 1)
            d[2, x, s] = np.min([d[2, x, s] + amount, np.ones_like(d[2, x, s], dtype=np.float32)], axis=0) * (2 if i % 3 == 2 else 1)

        sim.set_boundary(box)    
        if flowMode.current == 0:  
            sim.set_velocity(np.s_[0, box], 0)
            sim.set_velocity(np.s_[1, box], 0)
            sim.set_velocity(np.s_[0, -10:,:], speedOption.current)
            sim.set_velocity(np.s_[0,:10,:], speedOption.current)    
        elif flowMode.current == 1:
            xs, ys = np.meshgrid(np.arange(sim_shape[1]), np.arange(sim_shape[0]))
            r = np.sqrt((xs - sim_shape[1]//2)**2 + (ys - sim_shape[0]//2)**2)
            T = np.logical_and(r > sim_shape[1] // 2 * 0.3, np.abs(ys - sim_shape[0]//2) < 10)
            a = np.arctan2(xs[T] - sim_shape[1]//2, ys[T] - sim_shape[0]//2) + np.pi/2
            sim.set_velocity(np.s_[0, T], np.cos(a) * r[T] / 100 * speedOption.current)
            sim.set_velocity(np.s_[1, T], np.sin(a) * r[T] / 100 * speedOption.current)                     
        sim.step(dt, d)

        # render
        if modeOption.current == 0:
            sim_render = np.array(cv2.resize(np.clip(d.T * 255, 0, 255), (width, height)), dtype=np.uint8)            
        elif modeOption.current == 1:
            rgb = util.velocity_field_to_RGB(sim.get_velocity(), 0.25) # should be 0.5 (i.e. square root), but this shows the lower velocities better
            sim_render = np.array(cv2.resize(rgb.T * 255, (width, height)), dtype=np.uint8)    

        
        key = cv2.waitKey(int(max(1, 33 - dt * 1000))) & 0xFF
                         
        for option in options:
            option.poll_for_key(key)
                         
        if key == ord('q'):
            while cv2.waitKey(100) > 0:
                None
            break
        elif key == ord('r'):
            sim.reset()
            d[:] = 0
             
        if bgOption.current == 0:
            output = cv2.subtract(camera.masked_frame, sim_render)
        else:
            output = cv2.add(camera.masked_frame, sim_render)
        
        text_color = (0, 0, 0) if bgOption.current == 0 else (255, 255, 255)
        
        pos = np.array((30,30))
        for option in options:
            cv2.putText(output, str(option), tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, 0.25, text_color)
            pos = pos + [0,20]
            
        cv2.putText(output, '{:.2f}fps {:.2f}ms'.format(fps, 1000/fps), (510,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
            
        cv2.putText(output, 'q=quit, r=reset', (30,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
        cv2.imshow('window', output)
            
else:
    print("ERROR: Couldn't capture frame. Is Webcam available/enabled?")
        
# release the capture decive
# cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)