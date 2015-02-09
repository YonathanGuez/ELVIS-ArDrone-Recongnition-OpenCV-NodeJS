#include "Image.h"
#include <stdlib.h>
#include<iostream>
#include <stdio.h>
#include <iostream>
#include "opencv2\core\core.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"//bibliothèque de findContours et drawContours
#include "opencv2\calib3d\calib3d.hpp"
#include "opencv2\nonfree\features2d.hpp"
#include "opencv2\nonfree\nonfree.hpp"
#include "opencv2\imgproc\imgproc_c.h"
#include "opencv2\core\operations.hpp"
#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
#include <cmath>

#define PI 3.14
#define SEUIL 50
using namespace std;
using namespace cv;

int main(int argc,char* argv[])
{
	Image image;
	char touchesortie='n';//touche de sortie
	cout<<"//////////////// Sujet :Piloter un Drone ////////////////"<<endl<<endl<<endl;
	do
	{
		image.choix(image);
		cout<<endl<<endl;
		cout<<"Si vous desirez Finir le programme Taper: q "<<endl;
		cout<<"Sinon Taper: une autre touche  "<<endl;
		cin>>touchesortie;
		cout<< endl;
	}
	while(touchesortie!='q');
	system("pause");
	return 0;
}