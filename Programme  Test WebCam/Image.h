#ifndef IMAGE_H
#define IMAGE_H
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

class Image
{
public :
	Mat Orig;
	Image();
	char MenuVideo();
	void choix(Image image);
	void FonctionSurf();
	void FonctionForme();
	void FonctionCouleur();
	void FonctionCorrelation();
	void FonctionSurfModiF();
	void FonctionCouleurFleche();
	void FonctionCouleurFlecheCam();
	int PrintAngle1(Point2f C,Point2f F);
};

#endif