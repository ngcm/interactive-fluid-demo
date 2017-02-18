# Fluid sim demo prototype

import numpy as np
import cv2

import FluidSimModel

cv2.namedWindow("window", flags=cv2.WND_PROP_FULLSCREEN)
cap = cv2.VideoCapture(0)

ret, frame = cap.read()


if(ret):
    # sub-sample the webcam image to fit the fluid sim resolution
    step = 4

    # initialise the fluid sim arrays
    height, width, nc = np.shape(frame)
    width = int(width/step)
    height = int(height/step)
    u = np.zeros((width, height)) # x-velocity
    v = np.zeros((width, height)) # y-velocity
    u_prev = np.zeros((width, height)) # applied x-velocity
    v_prev = np.zeros((width, height)) # applied y-velocity
    d = np.zeros((width, height)) # smoke density
    d_prev = np.zeros((width, height)) # add smoke

    # fluid parameters
    diff = 0.001
    visc = 0.001
    dt = 0.0001

    while(True):    
        # Capture webcam frame
        ret, frame = cap.read()

        # convert to 2 colours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = -gray
        gray[gray < 150] = 0
        gray[gray > 0] = 255
            
        for i in range(100):
            gray[240-i/2:240+i/2,200+i] = 255
                
        cv2.circle(gray, (400, 100), 20, (255), 50)
         

        # copy the webcam data into the boundary
        box = np.array(gray[::step,::step].T, dtype=bool)
        # add the bounding box
        box[:, :5] = True
        box[:5, :] = True
        box[:, -5:] = True
        box[-5:, :] = True

        # add the smoke trails and wind

        '''
        d_prev[10:30, 10:20] = 100
        d_prev[10:30, 30:40] = 100
        d_prev[10:30, 50:60] = 100
        d_prev[10:30, 70:80] = 100
        d_prev[10:30, 90:100] = 100
        '''
        # clamp the density field
        
        d[5:6,:] = 0
        for i in range(10):
            x, y = 5, 10 + 10 * i
            d[x, y] = np.random.rand() * 255
             
        u[-20:-10,:] = 300
        u[10:20,:] = 300
        
        FluidSimModel.vel_step(u, v, u_prev, v_prev, visc, dt, box)
        FluidSimModel.dens_step(d, d_prev, u, v, diff, dt, box)   

        d[box] = 0
        d = np.clip(d, 0, 255)

        # resize the density field and overlay to the web cam image
        gwidth, gheight = np.shape(gray)
        resize = np.array(cv2.resize(d.T, (gheight, gwidth)), dtype=gray.dtype)
        cv2.addWeighted(resize, 0.8, gray, 0.2, 0, gray)

        # render
        cv2.imshow('window',gray)

        # quit, q, reset , r
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            u = np.zeros((width, height))
            v = np.zeros((width, height))
            d = np.zeros((width, height))
            u_prev = np.zeros((width, height))
            v_prev = np.zeros((width, height))
            d_prev = np.zeros((width, height))
            
else:
    print("ERROR: Couldn't capture frame. Is Webcam available/enabled?")
        
# release the capture decive
cap.release()
cv2.destroyAllWindows()

