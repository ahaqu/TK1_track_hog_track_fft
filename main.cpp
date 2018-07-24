// MyAlgorithmPlatform2.cpp : ¶¨Òå¿ØÖÆÌ¨Ó¦ÓÃ³ÌÐòµÄÈë¿Úµã¡£
//

// #include "../MyAlgorithm1/MyAlgorithm1.h"


#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <fstream>
#include <string>
#include "time.h"
#include <cstring>
//#include "windows.h"
#include "struct.h"
#include "CirculantTracker.h"
//#include "vld.h"
#include <vector>
#include <sstream>
#include "unistd.h"
#include<errno.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<string.h>
#include<netinet/in.h>
#include<netdb.h>
#include<unistd.h>
#include<arpa/inet.h>


//#define FFT
#define MOUSE_CONTROL

#define LOG if(1) cout
//#define TC if(0==1) 
#define TC //
#define PORTNUM 8888
#define RECV_PORT 8887
//#define IP_ADDRESS "192.168.106.123"
#define window_x 960
#define window_y 540
char *VIDEO_NAME;
char IP_ADDRESS[16];

int frameWidth;
int frameHeight;

int camera_video_type;


//#define VIDEO_NAME "cut_09_540.mp4"
//#define VIDEO_OUTPUT

#define CAMERA_VIDEO
#define CAMERA 1
#define VIDEO 2

#define LINUX

//#include <conio.h>
#include <queue>
#include <array>

#ifdef LINUX
#include "/usr/local/cuda-6.0/targets/armv7-linux-gnueabihf/include/cuda.h" 
#include "/usr/local/cuda-6.0/targets/armv7-linux-gnueabihf/include/cuda_runtime.h"
#include "/usr/local/cuda-6.0/targets/armv7-linux-gnueabihf/include/device_launch_parameters.h"

#else
#include "cuda.h" 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif
//#include "gpu_track.cuh"

#include <pthread.h>  

#ifdef CAMERA_VIDEO
#define BUFFER_SIZE 3
#define DROP_SIZE 10
bool paused = false;
bool quit = false;
#else
#define BUFFER_SIZE 3
#define DROP_SIZE 5
#endif
using namespace std;
using namespace cv;


int video_end = 1;
long count_read;
long count_process;
long count_write;

FloatRect initBB;
bool position_ready = true;
bool initBB_flag = true;

#ifdef MOUSE_CONTROL
FloatRect box;
bool drawing_box = false;
bool gotBB = false; // got tracking box or not
string video;
bool fromfile=false;

//FloatRect initBB;


pthread_mutex_t the_mutex;
//pthread_cond_t get_rectang, show_care;

int sockfd;
int new_fd;

void mouseHandler(int event, int x, int y, int flags, void *param);
void* tcp_recv(void *param);
void* udp_recv(void *param);
#endif

struct result_node_datatype{

	cv::Mat result;
	FloatRect track_loc;

	struct result_node_datatype* next;

};


typedef struct result_link_datatype{

	struct result_node_datatype *head;
	struct result_node_datatype *end;
	int result_num;


}result_link_type;

void result_push(result_link_type* result_link,result_node_datatype * result_node)
{
	if (result_link->head == NULL){

		result_link->head = result_node;
		result_link->end = result_link->head;
		result_link->result_num ++;


	}
	else
	{

		result_link->end->next = result_node;
		result_link->end = result_node;
		result_link->result_num ++;

	}

}


struct result_node_datatype* result_pop(result_link_type* result_link)
{
	struct result_node_datatype* tmp_node;
	if (result_link->head == NULL) return NULL;
	else if (result_link ->head == result_link->end)
	{
//		tmp_node = result_link->head;
//		result_link->head = result_link->end = tmp_node->next;
//		return tmp_node;
		return NULL;
	}
	else
	{
		tmp_node = result_link->head;
		result_link->head = result_link->head->next;
		result_link->result_num--;
		return tmp_node;
	}

}



static const int kLiveBoxWidth = 80;
static const int kLiveBoxHeight = 80;
void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour,2,7,0);
}


typedef struct {

	//´æ·ÅÖ¡Êý¾Ý
	Mat* array_frame;

	//´ý´¦ÀíÊý¾ÝÆì±ê
	int raw_flag;

	//´ÓÉãÏñÍ·È¡»ØÀ´µÄÊý¾Ý£¬´æµ½bufferµÄÓÎ±ê
	int prepare_flag;

	//³õÊ¼»¯±êÖ¾
	int init_flag;


} buffer_struct;

void * show_usage()
{

		cout<<"Usage: "<<endl;
		cout<<"1) use [camera] to read data from the camera "<<endl;
		cout<<"2) specify a file name to read data from a video file,like [cut01.mp4] "<<endl;
		cout<<"Sample: "<<endl;
#ifdef	FFT
		cout<<"track_fft camera"<<endl;
		cout<<"track_fft cut01.mp4"<<endl;
#else
		cout<<"track_hog camera"<<endl;
		cout<<"track_hog cut01.mp4"<<endl;
#endif
}

void * prepare_data(void* Param)
{
#ifdef CAMERA_VIDEO
	VideoCapture inputVideo;	

	if(!strcmp(VIDEO_NAME,"camera"))
	{
		inputVideo = VideoCapture(0);    //0ÎªÍâ²¿ÉãÏñÍ·µÄID£¬1Îª±Ê¼Ç±¾ÄÚÖÃÉãÏñÍ·µÄID
		camera_video_type = CAMERA;
	}
	else
	{
		inputVideo = VideoCapture(VIDEO_NAME);
		camera_video_type = VIDEO;
	}
	
	if(!inputVideo.isOpened())
	{
		cout<<"failed to open "<<VIDEO_NAME<<endl;
		quit = true;
		show_usage();
		exit(0);	
	}


	inputVideo.set(CV_CAP_PROP_FRAME_WIDTH, window_x);
        inputVideo.set(CV_CAP_PROP_FRAME_HEIGHT, window_y);

	frameWidth = inputVideo.get(CV_CAP_PROP_FRAME_WIDTH);
	printf(" Frame Width is %d\n", frameWidth);
	frameHeight = inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT);
	printf(" Frame Height is %d\n", frameHeight);
#else
#ifdef LINUX
	string sequenceBasePath = "/home/ubuntu/image";
	string imgFormat = sequenceBasePath+"/%d.jpg";
#else
	string sequenceBasePath = "C:\\Users\\Jack\\Desktop\\xbw\\code\\2015_0316_150447_004";

	string imgFormat = sequenceBasePath+"/img/%d.jpg";
#endif
	int startFrame = 2297;
	int endFrame = startFrame+BUFFER_SIZE;
#endif


//	VideoCapture inputVideo("C:\\Users\\Jack\\Desktop\\xbw\\code\\xbw_ok\\input.avi");


	buffer_struct* buffer = (buffer_struct*)Param;

	int frameInd = 0;
	char imgPath[256];
	Mat frame;
	while(1){

		if (buffer->init_flag == 0)
		{

			for(int i=0;i<BUFFER_SIZE;i++)
			{
		//		Mat frame;

#ifdef CAMERA_VIDEO
#ifdef FFT
				while(paused) ;
#endif
//				if(quit) exit(0);
				inputVideo >> frame;
#else
				frameInd = i+startFrame; 
				sprintf(imgPath, imgFormat.c_str(), frameInd);
				frame = cv::imread(imgPath, 0);
				if (frame.empty())
				{
					LOG << "error: could not read frame: " << imgPath << endl;
					LOG<< startFrame<<endl;
					return NULL;
				}
//				resize(frameOrig, frame, Size(1920, 1080));
#endif
				buffer->array_frame[i] = frame;
			}
			buffer->init_flag =1;
			buffer->prepare_flag = BUFFER_SIZE-1;

		}

#ifdef MOUSE_CONTROL
		else
#else
		else if(buffer->init_flag ==2)
#endif
		{
		//	Mat frame;
#ifdef LINUX
			while((buffer->raw_flag-buffer->prepare_flag + BUFFER_SIZE)%BUFFER_SIZE == 1) count_read++;
			buffer->prepare_flag =(buffer->prepare_flag+1)%BUFFER_SIZE;
//			buffer->prepare_flag = (buffer->raw_flag-1+BUFFER_SIZE)%BUFFER_SIZE;
#else
			while((buffer->raw_flag-buffer->prepare_flag + BUFFER_SIZE)%BUFFER_SIZE == 1);
			buffer->prepare_flag =(buffer->prepare_flag+1)%BUFFER_SIZE;
//			buffer->prepare_flag = (buffer->raw_flag-1+BUFFER_SIZE)%BUFFER_SIZE;
#endif			


			int index = buffer->prepare_flag;
/*
			buffer->prepare_flag = buffer->prepare_flag+1;
			buffer->prepare_flag = buffer->prepare_flag%BUFFER_SIZE;
*/
#ifdef CAMERA_VIDEO
			while(paused);
//			if(quit) exit(0);
			inputVideo >> frame;
			//usleep(20000);
			if (frame.empty())
			{
				LOG<<" video end "<<endl;
//				exit(0);
				quit = true;
				return NULL;
			}
#else
				if (frameInd ==0)
					frameInd = startFrame+BUFFER_SIZE;
				sprintf(imgPath, imgFormat.c_str(), frameInd);
//				Mat frameOrig = cv::imread(imgPath, 0);	
				frame = cv::imread(imgPath, 0);	
				if (frame.empty())
				{
					LOG << "error: could not read frame: " << imgPath << endl;
					LOG<< startFrame<<endl;
					return NULL;
				}
//				resize(frameOrig, frame, Size(1920, 1080));
				frameInd++;
#endif

			buffer->array_frame[index] = frame;
		}
		
#ifdef CAMERA_VIDEO
	}	
	return NULL;
#else
	}
	return NULL;
#endif

}


void* show_data(void* Param)
{
    // 压缩参数，jpeg
    vector<int> param = vector<int>(2);
    param[0] = CV_IMWRITE_JPEG_QUALITY;
    param[1] = 95; // default(95) 0-100

    Mat frame; // 存储每一帧图像
    vector<uchar> buff; // 存储压缩后的数据
    char s[20] = {0};

	result_link_type* result_link = (result_link_type*) Param;
#ifdef VIDEO_OUTPUT
	VideoWriter vwriter("output.avi",CV_FOURCC('M','J','P','G'),30,Size(960,540),1);
#endif

	bool setcallback_flag = false;

	struct result_node_datatype *result_node2;


	while(1){	

//		if(quit) exit(0);
		result_node2 = result_pop(result_link);

		if (result_node2==NULL) continue;

		rectangle(result_node2->result, result_node2->track_loc, CV_RGB(0, 255, 0));

#ifdef VIDEO_OUTPUT
		vwriter<<result_node2->result;
#else
#ifdef MOUSE_CONTROL
		//if(!position_ready || (position_ready && (result_link->result_num<=DROP_SIZE)))
		if(result_link->result_num<=DROP_SIZE)
#else
		if(result_link->result_num<=DROP_SIZE)
#endif
		{
			//int imgSize = result_node2->result.total()*result_node2->result.elemSize();
			//send(new_fd, result_node2->result.data, imgSize, 0);
			
			// 压缩图像
			imencode(".jpg", result_node2->result, buff, param);
			int bufflen = buff.size();
	            	// 发送数据的大小, 将二进制数据转换成文本再发送
	        	sprintf(s, "%d", bufflen);
		        send(new_fd, s, 15, 0);
	        	cout<<"bufflen = "<<s<<endl;
	        	// 发送
            		send(new_fd, &buff[0], buff.size(), 0);
			
//			imshow("result", result_node2->result);
			char key = waitKey(paused ? 0 : 1);

			switch (key)
			{
				case ' ':
					if (camera_video_type == VIDEO)
					{
						paused = !paused;
					}
/*					char key2 = waitKey();
					if (key2 == ' ')
					{
						paused == !paused;
					}
*/
					break;
				case 27:
					quit = !quit;
					break;
/*

				cout<< key <<endl;
				if (key == 27 || key == 113) // esc q
				{
					break;
				}
				else if (key == 112) // p
				{
					paused = !paused;
				}
				else if (key == 105)
				{
		//			doInitialise = true;
				}
*/
			}
//	CvCapture *capture = cvCaptureFromFile(VIDEO_NAME);
 //     IplImage* tmpIm_frame =  cvQueryFrame(capture);
 //     namedWindow("result", CV_WINDOW_AUTOSIZE);
//        cvShowImage("result",tmpIm_frame);
        //waitKey();
			
 //       		setMouseCallback("result", mouseHandler, NULL);
//			int key = waitKey(1000/25);
		}else{count_write++;}
#endif
#ifdef MOUSE_CONTROL
		if(!setcallback_flag)
		{
			setMouseCallback("result", mouseHandler, NULL);
			setcallback_flag = true;
		}
		else if (position_ready)
		{
			setMouseCallback("result",NULL,NULL);
		}
			
		
#endif
		result_node2->result.~Mat();
		free(result_node2);
	}
}



int main(int agrc, char *agrv[])
{
//	_tmain();
	if(agrc < 2)
	{
		show_usage();
		exit(0);	
	}
	VIDEO_NAME = agrv[1];
	if (VIDEO_NAME == NULL)
	{
		show_usage();
		exit(0);	
		
	}

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


	

//	sprintf(VIDEO_NAME,agrv[1]);
	
	count_read = count_process = count_write = 0;

	result_link_type *result_link = new result_link_type;
	result_link->head = result_link->end = NULL;
	result_link->result_num = 0;

	
	camera_video_type = 0;

#ifdef MOUSE_CONTROL
	position_ready = false;
#else
	position_ready = true;
#endif
	
	paused = false;
	quit = false;

#ifndef LINUX
	long t1=GetTickCount();
#endif

	string imgFormat;
#ifdef CAMERA_VIDEO
	frameWidth = 0;
	frameHeight = 0;

	float xmin = 460.0;
	float ymin = 300.0;
	float width = 40.0;
	float height = 88.0;

//	int frameWidth = 600;
//	int frameHeight = 480;

//	float xmin = 280.0;
//	float ymin = 100.0;
//	float width = 150.0;
//	float height = 300.0;
#else
	int frameWidth = 1920;
	int frameHeight = 1080;

	float xmin = 875.0;
	float ymin = 476.0;
	float width = 48.0;
	float height = 128.0;

#endif

#ifndef MOUSE_CONTROL

	initBB = FloatRect(xmin, ymin, width, height);
#endif
//	Mat result(frameHeight, frameWidth, CV_8UC3);

//	bool paused = false;

	/************************************************************************/
	/* loop starts                                                                     */
	/************************************************************************/
	float output_sigma_factor = 0.1;
#ifdef FFT
	float sigma = 0.2;
	float interp_factor = 0.075;
#else
	float sigma = 0.5;
	float interp_factor = 0.02;
#endif
	float lambda = 1e-4;
	float padding = 1.5;

	CirculantTracker tracker(output_sigma_factor,sigma, lambda, interp_factor,padding ,256);
	FloatRect track_loc=initBB;
#ifndef LINUX
	LOG<<"Before processing :"<< (GetTickCount()-t1) <<"ms"<<endl;
	t1=GetTickCount();
	long t3=GetTickCount();
	long t2;
#endif
	buffer_struct* buffer = new buffer_struct;
	buffer->array_frame = new Mat[BUFFER_SIZE];
	buffer->raw_flag = 0;
	buffer->prepare_flag=2;
	buffer->init_flag = 0;

	pthread_t pid;  
	pthread_create(&pid, NULL, prepare_data, buffer);  

	pthread_t pid_show;  
	pthread_create(&pid_show, NULL, show_data, result_link);  

	pthread_t pid_recv;
	pthread_create(&pid_recv, NULL, tcp_recv, NULL);


	//	prepare_data(buffer);

#ifdef CAMERA_VIDEO
#ifdef LINUX
//	sleep(2);
	while(!(buffer->init_flag)) ;

#else
//	Sleep(2000);
	while(!(buffer->init_flag)) ;

#endif
#else
#ifdef LINUX
//	sleep(2);
	while(!(buffer->init_flag)) ;

#else
//	Sleep(200);
	while(!(buffer->init_flag)) ;

#endif
#endif
//	printf("OK");

	unsigned char *tmpImageData = new unsigned char[frameWidth*frameHeight];

	while(1)
	{

//		LOG<<"count_read = "<<count_read<<";count_process = "<<count_process<<"; count_write = "<<count_write<<endl;
#ifdef CAMERA_VIDEO
		
		while(paused);
		if(quit)
		{
			usleep(1000000);
			 exit(0);
		}
#endif

#ifdef LINUX
		double fps_t1 = (double )cv::getTickCount();
#endif
		struct result_node_datatype *result_node = new struct result_node_datatype;
		result_node->next=NULL;

#ifndef LINUX
		t1=GetTickCount();
		t2=GetTickCount();
#endif
//		int index_frame = frameInd - startFrame;



		int index = buffer->raw_flag;
#ifdef LINUX
		while(1)
		{
			if(( buffer->prepare_flag - index +BUFFER_SIZE)%BUFFER_SIZE>1)
				break;

			if(quit)
			{
				usleep(1000000);
				 exit(0);
			}
			count_process++;
		}
#else
		while(1)
		{
			if(( buffer->prepare_flag - index +BUFFER_SIZE)%BUFFER_SIZE>1)
				break;
		}
/*
			if(( buffer->prepare_flag - index +BUFFER_SIZE)%BUFFER_SIZE>1)
				Sleep(500);
*/

#endif
#ifdef LINUX
		//printf(" index = %d\n", index);
//		cout<<"index = "<<index;
		
//		printf(" buffer->prepare_flag = %d \n",buffer->prepare_flag);	
//		printf("good,%d\n",((buffer->prepare_flag - index +BUFFER_SIZE)%BUFFER_SIZE));
#endif

//		Mat result = buffer->array_frame[buffer->raw_flag];

//		buffer->raw_flag = buffer->raw_flag+1;
//		buffer->raw_flag = (buffer->raw_flag)%BUFFER_SIZE;


#ifdef CAMERA_VIDEO
		Mat result = buffer->array_frame[index];
		Mat frame;

		if(!result.empty()){
			cvtColor(result, frame, CV_BGR2GRAY);
		}
		else
		{
			LOG<<"result is empty!"<<endl;
//			LOG<<"result."<<endl;
//			usleep(2000);
			result = buffer->array_frame[index+1];	
			cvtColor(result, frame, CV_BGR2GRAY);
			exit(0);
	//		exit(0);
//			continue;
		}

//		frame = result.clone();

#else
		Mat frame = buffer->array_frame[index];

		Mat result(frameHeight, frameWidth, CV_8UC3);

		cvtColor(frame, result, CV_GRAY2RGB);
#endif
		//cout<<"before ready"<<endl;
#ifdef MOUSE_CONTROL
		if(position_ready){
#endif
		//cout<<"ready"<<endl;
		for (int i=0; i<frameHeight; i++)
		{
#ifdef CAMERA_VIDEO
			unsigned char *AA=frame.ptr(i);
#else
			unsigned char* AA = frame.ptr(i);
#endif
			for (int j=0; j<frameWidth; j++)
			{
				tmpImageData[i*frameWidth+j] = *(AA + j);
			}
		}

//		LOG << "Track frame:" <<<< endl;
#ifndef LINUX
		long start_time = GetTickCount();
#endif
		if (buffer->init_flag == 1)
		{
			buffer->init_flag =2;
			// ¿ªÊ¼³õÊ¼»¯

#ifdef FFT
			tracker.initialize(tmpImageData,frameWidth,frameHeight,int(initBB.m_xMin),int(initBB.m_yMin),int(initBB.m_width),int(initBB.m_height));

#else
			tracker.initialize(tmpImageData,frameWidth,frameHeight,int(initBB.m_xMin),int(initBB.m_yMin),int(initBB.m_width),int(initBB.m_height),4);
#endif
//			LOG << "Track Init Complete time:" <<(GetTickCount()-start_time)<< endl;

		}
		else
		{
#ifdef LINUX
			TC long t5 = clock();
			//cout<< "in else"<<endl;
#else
			start_time = GetTickCount();
#endif
			tracker.performTracking( tmpImageData) ;
			track_loc = tracker.getTargetLocation();
#ifdef LINUX
//			LOG<<"Track per frame time = "<<clock()-t5<<endl;
#else
			LOG << "Track time per frame:" <<GetTickCount()-start_time<< endl;
#endif
		}

//		LOG<< " Track time per frame (my own): " << GetTickCount()-t1<<"clocks" << endl;
//		t1=GetTickCount();
#ifdef MOUSE_CONTROL
		}
#endif	

		buffer->raw_flag = buffer->raw_flag+1;
		buffer->raw_flag = (buffer->raw_flag)%BUFFER_SIZE;
#ifdef MOUSE_CONTROL
		if (!position_ready) 
			usleep(30000);
#endif

		result_node->result = result;
		result_node->track_loc = track_loc;
		result_push(result_link,result_node);

		frame.~Mat();
#ifdef LINUX
		fps_t1 = (( double)cv::getTickCount() - fps_t1) / cv::getTickFrequency();
	        double fps = 1.0/fps_t1;
//        cout<<"after fps = "<<fps<<endl;
//       cout<<"\n\n\n\n"<<endl;
#endif



//		LOG<< "After tracking, some ending operations costs: " << GetTickCount()-t1<<"clocks" << endl;
//		Sleep(60);

		//------------------------------------KCF Tracking½áÊø-------------------------------------


/*
		if (key != -1)
		{
			if (key == 27 || key == 113) // esc q
			{
				break;
			}
			else if (key == 112) // p
			{
				paused = !paused;
			}
			else if (key == 105)
			{
				doInitialise = true;
			}
		}
*/
		/*
		if ( index_frame == 400)
		{
			LOG << "\n\nend of sequence, press any key to exit" << endl;
			//				waitKey();
		}
		*/

//		LOG<< "Every Frame cost: " << GetTickCount()-t2 << endl;

	}

	waitKey();
	delete []tmpImageData;


	pthread_join(pid,NULL);
	pthread_join(pid_show,NULL);
	pthread_join(pid_recv,NULL);

	return EXIT_SUCCESS;



	return 0;
}
#ifdef MOUSE_CONTROL
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

#endif


void* tcp_recv(void *param)
{
	
	printf("tcp_recv -----\n");
	
	while(initBB_flag)
//	while(1)
	{
		recv(new_fd, (void*)&initBB, size_t(sizeof(initBB)), 0);
	//	ss = recv(new_fd, buffer, sizeof(buffer), 0);
	//	printf("buffer = %s\n",buffer);
	//	printf(" ss = %d\n",ss);
		printf("Initial Tracking Box = x:%f y:%f h:%f w:%f\n", initBB.m_xMin, initBB.m_yMin, initBB.m_width, initBB.m_height);
		initBB_flag = false;
		position_ready = true;
	}
	
//	close(new_fd);
	return NULL;
}
