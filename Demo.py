# Fluid sim demo prototype

import time
import numpy as np
import cv2

import StamFluidSim
import util

modes = ['smoke', 'velocity']
mode = 0

bgmodes = ['white', 'black']
bgmode = 0

speed = 300

cv2.namedWindow("window", flags=cv2.WND_PROP_FULLSCREEN)
cap = cv2.VideoCapture(0)

ret, input_frame = cap.read()
input_frame = cv2.flip(input_frame, 1)

ltime = time.time()
if(ret):
    # sub-sample the webcam image to fit the fluid sim resolution
    step = 4

    # initialise the fluid sim arrays
    height, width, nc = np.shape(input_frame)    
    sim_shape = (int(width/step), int(height/step))
    diff = 0.001
    visc = 0.0001
    sim = StamFluidSim.StamFluidSim(sim_shape, diff, visc)
    
    d = np.zeros((3, *sim_shape), dtype=np.uint8)
    
    dt = 0.0001
    
    level = 150

    while(True):    
        # Capture webcam frame
        ret, input_frame = cap.read()
        input_frame = cv2.flip(input_frame, 1)

        # convert to 2 colours
        mask = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        if bgmode == 0:
            mask = -mask
        mask[mask < level] = 0
        mask[mask > 0] = 255
            
        #mask[:] = 0
        '''
        for i in range(100):
            mask[150+i:152+i,150+i:153+i] = 255
            mask[210+i:212+i,150+i:153+i] = 255
                
        cv2.circle(mask, (400, 100), 20, (255), 50)
        cv2.rectangle(mask,(370, 310), (430, 370), (255), -50)
        '''
        # copy the webcam data into the boundary
        box = np.array(mask[::step,::step].T, dtype=bool)

        # add the bounding box
        box[:, :2] = True
        box[:, -2:] = True

        sim.set_boundary(box)
        masked_input_frame = np.zeros_like(input_frame)
        if bgmode == 0:
            masked_input_frame[:,:] = 255
        masked_input_frame = cv2.add(input_frame, 0, dst=masked_input_frame, mask=mask)

        smoke_amount = 1
        for i in range(10, height, 10):
            x, y = 3, i
            d[0, x, y-smoke_amount:y+smoke_amount] = (np.random.rand() * 64 + 64) * (2 if i % 3 == 0 else 1)
            d[1, x, y-smoke_amount:y+smoke_amount] = (np.random.rand() * 64 + 64) * (2 if i % 3 == 1 else 1)
            d[2, x, y-smoke_amount:y+smoke_amount] = (np.random.rand() * 64 + 64) * (2 if i % 3 == 2 else 1)
             
        sim.set_velocity(np.s_[0, -20:-10,:], speed)
        sim.set_velocity(np.s_[0,10:20,:], speed)                
        sim.step(dt, d)
        d = np.clip(d, 0, 255)

        # render
        if mode == 0:
            sim_render = np.array(cv2.resize(d.T, (width, height)), dtype=np.uint8)            
        elif mode == 1:
            rgb = util.velocity_field_to_RGB(sim.get_velocity())
            sim_render = np.array(cv2.resize(rgb.T * 255, (width, height)), dtype=np.uint8)    
            
        curtime = time.time()
        fps = 1 / (curtime - ltime)
        ltime = curtime
            
        if bgmode == 0:
            masked_input_frame = cv2.subtract(masked_input_frame, sim_render)
        else:
            cv2.addWeighted(sim_render, 1, masked_input_frame, 1, 0, masked_input_frame)
        
        text_color = (0, 0, 0) if bgmode == 0 else (255, 255, 255)
        
        cv2.putText(masked_input_frame, 'mode={}'.format(modes[mode]), (30,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
        cv2.putText(masked_input_frame, 'bgmode={}'.format(bgmodes[bgmode]), (30,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
        cv2.putText(masked_input_frame, 'speed={}'.format(speed), (30,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
        cv2.putText(masked_input_frame, 'fps={:.2f}'.format(fps), (30,90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
        cv2.putText(masked_input_frame, 'q=quit, r=reset, m=mode, b=bg, []=level, -+=speed', (30,460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
        cv2.imshow('window', masked_input_frame)

        # quit, q, reset , r
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            sim.reset()
            d[:] = 0
        elif key == ord('m'):
            mode = (mode + 1) % len(modes)
        elif key == ord('['):
            level = max(level - 10, 10)
        elif key == ord(']'):
            level = min(level + 10, 250)
        elif key == ord('-'):
            speed = max(speed - 20, 00)
        elif key == ord('='):
            speed = min(speed + 20, 1000)
        elif key == ord('b'):
            bgmode = (bgmode + 1) % len(bgmodes)
            
else:
    print("ERROR: Couldn't capture frame. Is Webcam available/enabled?")
        
# release the capture decive
# cap.release()
cv2.destroyAllWindows()

