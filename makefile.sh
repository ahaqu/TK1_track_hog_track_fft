rm track_fft
rm track_hog
echo "rm track_fft"
echo "rm track_hog"
nvcc -c gpu_track.cu
echo "nvcc -c gpu_track.cu"
#g++ -c CirculantTracker.cpp GeneralFFT_2D.o `pkg-config --libs opencv` `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib
g++ -c CirculantTracker.cpp `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib
echo "g++ -c CirculantTracker.cpp `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib"
echo
g++ main.cpp CirculantTracker.o gpu_track.o GeneralFFT_2D.o GeneralFFT_1D.o `pkg-config --libs opencv` `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib -std=c++11 -o track_hog
echo "g++ main.cpp CirculantTracker.o gpu_track.o GeneralFFT_2D.o GeneralFFT_1D.o `pkg-config --libs opencv` `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib -std=c++11 -o track_hog"
echo

#g++ -c CirculantTracker.cpp GeneralFFT_2D.o `pkg-config --libs opencv` `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib -o CirculantTracker_fft.o -D FFT
g++ -c CirculantTracker.cpp `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib -o CirculantTracker_fft.o -D FFT
echo "g++ -c CirculantTracker.cpp `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib -o CirculantTracker_fft.o -D FFT"
echo
g++ main.cpp CirculantTracker_fft.o gpu_track.o GeneralFFT_2D.o GeneralFFT_1D.o `pkg-config --libs opencv` `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib -std=c++11 -o track_fft -D FFT
echo "g++ main.cpp CirculantTracker_fft.o gpu_track.o GeneralFFT_2D.o GeneralFFT_1D.o `pkg-config --libs opencv` `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib -std=c++11 -o track_fft -D FFT"
echo 
echo "-----------------------SUCCESS!!!"
echo
##./a.out
