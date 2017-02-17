# interactive-fluid-demo

The aim of this project is to create an engaging, interactive demonstration of the kinds of simulations we develop as part of the [NGCM-CDT](http://ngcm.soton.ac.uk/). The basic concept is to combine real-time simulations with live video so that participants can interact with the simulation directly. It was inspired by the Lego(tm) Fluid Dynamics idea suggested at the 2017 NGCM outreach workshop. The ultimate goal is to create an easy to setup, maintain and develop demo that can be an ongoing part of the NGCM's contribution to the University of Southampton's public outreach (http://www.southampton.ac.uk/per/).

As a proof-of-concept, I have combined a simple real-time 2D fluid dynamics simulation with live webcam footage, using *Python* and a computer vision library, *OpenCV*. This can be interacted with directly and simultaneously used to examine the aerodynamics of any object (like a car built from Lego(tm)) place in front of the camera.

Planned improvements;
* The model is very rough, and does not handle boundaries very well.
* Replace the model entirely (SPH), as this one tends to soften too much.
* Could have multiple rendering modes (e.g. calibration, velocity field)
* Improve the subject/background separation, perhaps using *OpenCV*'s image processing (chromakey?)

## OpenCV

*OpenCV* is an open source computer vision library. At present it's only used to get data from the webcam, superimpose some images, and render an output window. It is also a little tricky to install. An alternative may be warranted if this makes it difficult for people to collaborate. I'm sticking with it for now as I have not found a suitable alternative and the computer vision, image processing and GUI functions may prove useful to the project as it develops.

### Installation via pre-built packages (Not working?)

I wasn't able to get a fully working python package. The *OpenCV* packages available on pip are apparently very out-of-date, so that may be the problem, but I think the problem is with *ffmpeg*. These packages looked promising, but no luck with them so far;
```
conda install -c menpo ffmpeg
conda install -c menpo opencv
```

### Build from source

To get *OpenCV* working, including video support (*ffmpeg*) I had to compile from source.

Installation steps derived from [pyimagesearch blog](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/). I've avoiding using environments, which means less steps total, but a slightly more involved *cmake* step.

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

## VirtualBox

I currently develop inside a VM. If doing the same, make sure to install the [VirtualBox 5.1.14 Oracle VM VirtualBox Extension Pack](https://www.virtualbox.org/wiki/Downloads) to get access to the host webcam from the guest VM. Then you should be able to enable the webcam from the `Devices` menu.