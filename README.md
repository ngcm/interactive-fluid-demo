# interactive-fluid-demo

In the [NGCM-CDT](http://ngcm.soton.ac.uk/), we spend a lot of time programming complex simulations designed to be run on powerful computers. But you don't need a fast computer or a large, complicated program to model real physics. For example, this fluid physics simulation is only about 150 lines of Python code and can be run in real time on a modest single core computer.

## Some Photos From The Recent [SOTSEF](https://www.facebook.com/sotsef) Event

![wind tunnel mode](https://github.com/ngcm/interactive-fluid-demo/blob/master/virtualwindtunnel.jpeg)
![interactive mode](https://github.com/ngcm/interactive-fluid-demo/blob/master/interactive.jpeg)

## Project Aim

The aim of this project is to create an engaging, interactive demonstration of the kinds of simulations we develop as part of the [NGCM-CDT](http://ngcm.soton.ac.uk/). The basic concept is to combine real-time simulations with live video so that participants can interact with the simulation directly. It was inspired by the Lego(tm) Fluid Dynamics idea suggested at the 2017 NGCM outreach workshop. The ultimate goal is to create an easy to setup, maintain and develop demo that can be an ongoing part of the NGCM's contribution to the University of Southampton's public outreach (http://www.southampton.ac.uk/per/).

As a proof-of-concept, I have combined a simple real-time 2D fluid dynamics simulation with live webcam footage, using *Python* and a computer vision library, *OpenCV*. This can be interacted with directly and simultaneously used to examine the aerodynamics of any object (like a car built from Lego(tm)) placed in front of the camera.

## Click for Video
[![FluidDemo](https://github.com/ngcm/interactive-fluid-demo/blob/master/FluidDemo.png)](https://raw.githubusercontent.com/ngcm/interactive-fluid-demo/master/FluidDemo.webm)

## Dependencies

### OpenCV

**OpenCV** is an open source computer vision library with a **Python** interface. When this project started, **OpenCV** was just a way to get web-cam data into **Python**. However, as the project has progressed, it has made greater and greater use of the this library's functions; to resize and recolour images, to identify foreground object in the webcam image with *Background Subtraction*, and to measure objects velocities with *Optical Flow*.

So, now the project is irrecoverably tied to **OpenCV**, and this is a problem because it remains an awkward library to install.

### Pynput

This project now uses **pynput** to capture keyboard key presses. **pynput** lets you register a callback to be triggered whenever a key is pressed, thus no key presses are missed and there is no pause in the update. This is big improvement over **OpenCV**s ****`waitkey()` function which polls for key presses during a mandatory pause (too short and you miss the key, too long the the frame rate is noticeably affected).

### Numba

For a trivial to implement speed boost, this project uses the JIT complier package Numba.

### Cython

While the project is mostly written in Python, it uses Cython to interface with the simulations compiled shared object written in C.

### Numpy

Numpy is used extensively, whenever performaing array operations. Note also that OpenCV depends directly on Numpy.

## Installation

### Ubuntu 16.04 + Pre-built packages (Recommended)

There are OpenCV Python packages in a third party channel, "menpo", that work well in Ubuntu (or Lubuntu, etc.) 16.04. So, if you are using this OS or are willing to set-up a VM, then this is by far the easiest method to get this project up and running.

This project is quite a performance hog, so a VM inevitably limits your frame rate/resolution. Also, if using a VM make sure to install the [VirtualBox Extension Pack](https://www.virtualbox.org/wiki/Downloads) to get access to the host webcam from the guest VM. Then you should be able to enable the webcam from the `Devices` menu.

Once you have your OS up and running, you'll want to create an environment for OpenCV to avoid package conflicts. I recommend [Conda](https://conda.io/docs/download.html) for this. Once Conda is installed, run the following commands in the terminal;

```
$ conda create -n opencv -c menpo opencv3 numba cython
```

Activate the environment and then install **pynput** as that's not available via Conda;
```
$ source activate opencv
$ pip install pynput
```

That's your environment setup. Now you just need to get the source for this project;
```
$ git clone https://github.com/ngcm/interactive-fluid-demo.git
```
Change into the project folder and build the C coded simulation shared object;
```
$ cd interactive-fluid-demo
$ make
```
Now the library is built, you can run the simulation any time after that by calling;
```
python run.py
```

### MacOS (Currently Not Working)

*Note that as of early March 2017, this method no longer works. It's not clear if that's because the project is using more of OpenCV or the available packages have changed.*

If you are on a Mac, then the conda-forge channel has working packages (thanks [ryanpepper](https://github.com/ryanpepper) for discovering this). Again, Conda is recommended for creating an conflict free environment. Once that is installed, run the following;
```
$ conda create -n opencv -c conda-forge opencv numba cython
```
Then continue with the installation instructions for Ubuntu.

### Building OpenCV from source

If working pre-built packages aren't available, then you'll need to build OpenCV from source.

*These installation steps derived from [pyimagesearch blog](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/). I've avoided using environments, which means less steps total, but a slightly more involved `cmake` step*

* Grab all the dependencies (if a particular version doesn't exist, try removing the version number, e.g. `libpng12-dev` > `libpng-dev`);
  ```
  sudo apt-get install build-essential cmake pkg-config libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python2.7-dev python3.5-dev
  ```

* Grab the source for *OpenCV* and unzip (n.b. version may have been updated, so modify the url accordingly);
  ```
  wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.2.0.zip
  unzip opencv.zip
  ```

* *This step probably isn't necessary*. This is the source for the additional "non-free" (for commercial use) parts of the library, which this project doesn't use and probably won't. It shouldn't prevent building *OpenCV* if left out. Grab the source and unzip (n.b. again version may have been updated);
  ```
  wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.2.0.zip
  unzip opencv_contrib.zip`
  ```

* Install your *Python* of choice (rest of steps assume *Python 3*) and install *numpy*;
  ```
  pip3 install numpy
  ```

* Now you should be ready to build the make file, using *cmake*. Note all of the compilation flags, especially the paths to the python executables/libraries/includes, make sure they are correct for your system. You may also run into an issue with `stdlib.h`, in which case add the flag `-D ENABLE_PRECOMPILED_HEADERS=OFF`;
  ```
  cd opencv-3.2.0/
  mkdir build
  cd build
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.2.0/modules \
      -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
      -D PYTHON3_LIBRARIES=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.5/dist-packages/numpy/core/include \
      -D PYTHON3_PACKAGES_PATH=/usr/local/lib/ \
      -D BUILD_EXAMPLES=ON ..
  ```

* When this completes you'll see a summary of all the options. Scroll up and double check that the *Python* version you want to use has the correct `Interpreter`, `Libraries`, `numpy`, and `packages path`. If these are not correct you won't build the *Python* bindings.

* Assuming the options are correct, build and (assuming the build reached 100%) install;
  ```
  make
  make install
  ```

* Now track down the shared library, `cv.so`, you just built. For me it ended up in `/usr/local/lib/python3.5/dist-packages/`, so start there. It needs to be somewhere *Python* can find it. That should be somewhere around there, but I could not determine the right place, so I moved it to the home directory as that is where I would run *Python*.

* Test the steps were successful by trying to import the package in *Python*;
  ```
  import cv2
  ```

## Configuration

It's possible to configure most of the simulations run-time options (e.g. flow-direction, sim resolution, etc.) before running the program by editing the `configuration.py` file. There you can also select between the C and Python/numpy versions of the sim (Note the Python version of the sim is much slower, it may however be useful for debugging purposes).

## How to run

If you followed the installation steps in this readme, then make sure you are in the Opencv Conda environment you created `source activate opencv`. Then run the program by calling `python run.py`.

While the program is running, press the <kbd>f</kbd> key to switch between full-screen and windowed mode, and <kbd>d</kbd> key to switch between normal and debug modes, where you will also find a on-screen list of the other controls.

### Foreground/Background Separation

The program implements a couple of different methods for separating background (which defines the fluid volume) and foreground (which defines the solid boundaries). Switch between these using the <kbd>b</kbd> key.

* *'white'*: this mode separates out the white parts of the image as background by selecting particular low *saturation* and high *value* values from the HSV colour space. You can alter the ranges using the <kbd>1</kbd> & <kbd>2</kbd> keys for the *value* and <kbd>3</kbd> & <kbd>4</kbd> keys for the *saturation*.

* *'black'*: Similar to 'white', but this time separating out the black parts of the image, i.e. the pixels with low *value* values from the HSV colour space. You can alter the range using the <kbd>1</kbd> & <kbd>2</kbd> keys.

* *'hue'*: This mode separates out pixels with particular *hues* and *values*. You can alter the ranges using the <kbd>1</kbd> & <kbd>2</kbd> keys for the *value* and <kbd>3</kbd> & <kbd>4</kbd> keys for the *hue*.

* *'bg_subtract'*: This mode uses one of OpenCV's [*background subtraction*](https://en.wikipedia.org/wiki/Background_subtraction) method, `BackgroundSubtractorKNN`. It currently uses a fixed learning rate, but can be altered in the `util/Camera.py` file.

### Optical Flow

To measure foreground object velocities, this project utilises another of OpenCV's methods, `calcOpticalFlowFarneback`, an [*Optical Flow*](https://en.wikipedia.org/wiki/Optical_flow) algorithm. This is on by default, but it slows down the simulation while simultaneously requiring a high frame-rate to function well. It can be toggled with the <kbd>o</kbd> key or in advance in the `configuration.py` file.
