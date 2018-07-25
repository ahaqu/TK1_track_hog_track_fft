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
Run:

$ ./image-stitching <file1> <file2> ...

The output file is out.jpg. You can play with the example data to start with.

Before dealing with very large images (4 megapixels or more), it's better to resize them. (I might add this feature in the future)

In cylinder/translation mode, the input file names need to have the correct order.
Examples (All original data available for download):

Zijing Apartment in Tsinghua University: dorm

"Myselves": myself
Carnegie Mellon University from 38 images 
