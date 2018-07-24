##g++ client_frame.cpp -o client_frame `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib
g++ -o recv_video recv_video.cpp -I /usr/local/include/opencv -L /usr/local/lib -lopencv_core -lopencv_highgui -lpthread -lrt  `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib

