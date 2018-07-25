# TK1_track_hog_track_fft
OpenPano is a panorama stitching program written in C++ from scratch (without any vision libraries). It mainly follows the routine described in the paper Automatic Panoramic Image Stitching using Invariant Features, which is also the one used by AutoStitch.

Build Status
Compile Dependencies:

    gcc >= 4.7 (Or VS2015)
    Eigen
    FLANN (already included in the repository, slightly modified)
    CImg (optional. already included in the repository)
    libjpeg (optional if you only work with png files)
    cmake or make

Eigen, CImg and FLANN are header-only, to simplify the compilation on different platforms. CImg and libjpeg are only used to read and write images, so you can easily get rid of them.

On ArchLinux, install dependencies by: sudo pacman -S gcc sed cmake make libjpeg eigen

On Ubuntu, install dependencies by: sudo apt install build-essential sed cmake libjpeg-dev libeigen3-dev
Compile:
