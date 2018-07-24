#include <stdio.h>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<errno.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<string.h>
#include<netinet/in.h>
#include<netdb.h>
#include<unistd.h>
#include<arpa/inet.h>
#include "struct.h"
#include <highgui.h>
using namespace cv;
using namespace std;

#define PORTNUM 8888
#define DATALEN 1024
#define window_x 640
#define window_y 480  

int video_end = 1;
long count_read;
long count_process;
long count_write;

FloatRect initBB;
bool position_ready = true;

FloatRect box;
bool drawing_box = false;
bool gotBB = false; // got tracking box or not
string video;
bool fromfile=false;
void mouseHandler(int event, int x, int y, int flags, void *param);
void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
        IntRect r(rRect);
        rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour,2,7,0);
}

int main(void)
{
        int sockfd,new_fd;
        struct sockaddr_in server_addr;
        struct sockaddr_in client_addr;
        //创建套接字
        sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if(sockfd == -1)
        {
                printf("socket error !");
                return 0;
        }
        //绑定IP和端口
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(PORTNUM);//端口8888
        server_addr.sin_addr.s_addr = INADDR_ANY;
        if(bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1)
        {
                printf("bind error !");
        }

        //开始监听
        if(listen(sockfd, 5) == -1)
        {
                printf("listen error !");
                return 0;
        }

        //循环接收数据
        socklen_t nAddrlen = sizeof(client_addr);

        printf("等待连接...\n");
        do
        {
                new_fd = accept(sockfd, (struct sockaddr *)&client_addr, &nAddrlen);
        }while(new_fd == -1);
        printf("接受到一个连接：%s \r\n", inet_ntoa(client_addr.sin_addr));

/*	Mat frame = Mat::zeros(window_x, window_y, CV_8UC3);
	int imgSize = frame.total()*frame.elemSize();
	cout<<"imgSize = "<<imgSize<<endl;
*/
//	Mat frame;
	cvNamedWindow("server", CV_WINDOW_AUTOSIZE);
	
//	int imgSize = frame.total()*frame.elemSize();
//	char buffer[imgSize];
	int bytes;

	cout<<"create frame"<<endl;
        while(true)
        {

		Mat frame = Mat::zeros(window_x, window_y, CV_8UC3);
		int imgSize = frame.total()*frame.elemSize();
		char buffer[imgSize];
		cout<<"imgSize = "<<imgSize<<endl;
		for (int i= 0; i< imgSize; i += bytes)
		{
			if((bytes = recv(new_fd, buffer + i, imgSize - i,0)) == -1)
			{
				cout<<"recv failed!!"<<endl;
		//		exit(0);
			}
			
			
		}
		Mat img(Size(window_x,window_y),CV_8UC3,buffer);

/*

		
		int ptr = 0;
		for(int i = 0; i< frame.rows; i++)
		{
			for(int j = 0; j< frame.cols; j++)
			{
				frame.at<cv:: Vec3b>(i,j) = cv::Vec3b(buffer[ptr+0],buffer[ptr+1],buffer[ptr+2]);
				ptr += 3;
			}
		}
*/
                //接收数据
                //recv(new_fd, buffer, size_t(sizeof(buffer)), 0);
		//frame(Size(window_y, window_x), CV_8UC3, buffer);
         /*       if(ret > 0)
                {
                        revData[ret] = 0x00;
                        for(i = 0; i < image_src->height; i++)
                        {
                                for (j = 0; j < image_src->width; j++)
                                {
                                        ((char *)(image_src->imageData + i * image_src->widthStep))[j] = revData[image_src->width * i + j];
                                }
                        }
                        ret = 0;
                }
	
               cvShowImage("server", image_src);
                cvWaitKey(1);
	*/
	//	cvWaitKey(10);
		setMouseCallback("server", mouseHandler, NULL);
		rectangle(img, initBB, Scalar(0,0,255));
		imshow("server", img);
		cvWaitKey(1);
		
		//usleep(100000);
        }
//        cvDestroyWindow("server");
        close(sockfd);
        return 0;
}


void mouseHandler(int event, int x, int y, int flags, void *param)
{
        switch (event)
        {
                case CV_EVENT_MOUSEMOVE:
                if (drawing_box)
                {
                        //box.width = x - box.x;
                        //box.height = y - box.y;

                        box.m_width = x - box.m_xMin;
                        box.m_height = y - box.m_yMin;
                }
                break;
                case CV_EVENT_LBUTTONDOWN:
                        drawing_box = true;
                        //box = IntRect(x, y, 0, 0);
                        box = FloatRect(x, y, 0, 0);
                break;
                case CV_EVENT_LBUTTONUP:
                        drawing_box = false;
                        if (box.m_width < 0)
                        {
                                box.m_xMin += box.m_width;
                                box.m_width *= -1;
                        }
                        if( box.m_height < 0 )
                        {
                                box.m_yMin += box.m_height;
                                box.m_height *= -1;
                        }

                        gotBB = true;
                        printf("Initial Tracking Box = x:%f y:%f h:%f w:%f\n", box.m_xMin, box.m_yMin, box.m_width, box.m_height);
                        initBB = FloatRect(box.m_xMin, box.m_yMin, box.m_width, box.m_height);
                        position_ready = true;
/*
#ifdef FFT
                        if(paused)
                                paused = true;
#endif
*/
                break;
                default:
                break;
        }
}

void *tcp_send(void *param)
{
	
}
