#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN || event == EVENT_MBUTTONDOWN )
    {
        cout << "Clicked - position (" << x << ", " << y << ")" << endl;
    }
}

int main(int argc, char** argv)
{
    // Read image from file 
    Mat img = imread("Dataset/00000-color.png");;

    //if fail to read the image
    if ( img.empty() ) 
    { 
        cout << "Error loading the image" << endl;
        return -1; 
    }

    //Create a window
    namedWindow("My Window", 1);

    //set the callback function for any mouse event
    setMouseCallback("My Window", CallBackFunc, NULL);

    //show the image
    imshow("My Window", img);

    // Wait until user press some key
    waitKey(0);

    return 0;
}