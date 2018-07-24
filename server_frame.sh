##g++ client_frame.cpp -o client_frame `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib
g++ -g -o server_frame server_frame.cpp -I /usr/local/include/opencv -L /usr/local/lib -lopencv_core -lopencv_highgui -lpthread -lrt  `pkg-config --cflags opencv` -L /usr/local/cuda-6.0/lib

