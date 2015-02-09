#include "Image.h"
#include <stdlib.h>
#include<iostream>
#include <stdio.h>
#include <iostream>
#include "opencv2\core\core.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
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

Image::Image()
{
	Orig=0;
}
char Image::MenuVideo()
{
	char k=0;
	cout<<" \t Les images sont saisir par une Camera "<<endl<<endl<<endl<<endl;
	cout<<"Quatre choix :Pour detecter une image: "<<endl<<endl;
	cout<<"\tTaper 1: Methode par detection de Couleur "<<endl<<endl;
	cout<<"\tTaper 2: Methode de detection de Forme "<<endl<<endl;
	cout<<"\tTaper 3: Methode par Correlation "<<endl<<endl;
	cout<<"\tTaper 4: Methode par Point d'interet "<<endl<<endl;
	cout<<"\tTaper 5: Etude par la couleur "<<endl<<endl;
	cout<<"\tTaper 6: Etude par Surf avec Detection d'angle "<<endl<<endl;
	cout<<"\tTaper 7: Etude par la couleur avec Webcam "<<endl<<endl;
	cin>>k;
	return k;
}
void Image::choix(Image image)
{
	char k=image.MenuVideo();
	if(k=='1')
	{
		image.FonctionCouleur();	
	}
	else if (k=='2')
	{
		image.FonctionForme();
	}
	else if(k=='3')
	{
		image.FonctionCorrelation();
	}
	else if(k=='4')
	{
		image.FonctionSurf();
	}
	else if(k=='5')
	{
		image.FonctionCouleurFleche();
	}
	else if(k=='6')
	{
		image.FonctionSurfModiF();
	}
	else if(k=='7')
	{
		image.FonctionCouleurFlecheCam();
	}

	else return choix(image);
}
//Fonction Donne l'angle sur 180° 
//Ne prend pas en compte qunad la fleche est ver le bas 
double PrintAngle(RotatedRect calcul)
{
	double resultat=0;
	if(calcul.size.width <calcul.size.height) 
	{
		return resultat=calcul.angle+90;
	}
	else
	{
		return resultat=calcul.angle+180;
	}	
}

//Fonction qui calcule l'angle de rotation 
//a effectuer pour avoir une fleche ver le haut (90°)
int Image::PrintAngle1(Point2f C,Point2f F)
{
	//on part du Principe que si la fleche est ver le haut a 90 degret alors le drone va tout droit
	double angle=0;
	angle = int(atan((C.y-F.y)/(C.x-F.x))*180/PI);
	if(F.x >= C.x)// La fleche est du coter gauche
	{
		cout<<"Ver la Gauche: "<<endl;
		angle=90-angle;
		cout<<"il faut tourner de "<<angle<<endl;
		return (int)angle;
	}
	else //la fleche est du coter Droite
	{
		cout<<"Ver la Droite : "<<endl;
		angle=90+angle;
		cout<<"il faut tourner de "<<angle<<endl;
		return (int)angle;
	}
}

void Image::FonctionSurf()
{
//INITIALISATION
    Mat object = imread( "fleche_devant1.png", CV_LOAD_IMAGE_GRAYSCALE );
	Mat des_object;
	VideoCapture cap(0);
	std::vector<KeyPoint> kp_object;
    FlannBasedMatcher matcher;
    SurfFeatureDetector detector(500);
	SurfDescriptorExtractor extractor;
	std::vector<Point2f> obj_corners(4);
    //Enregistrement des 4 coins de l'image 
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( object.cols, 0 );
    obj_corners[2] = cvPoint( object.cols, object.rows );
    obj_corners[3] = cvPoint( 0, object.rows );

	//Detection des points Clefs de l'image regerence
    detector.detect( object, kp_object );
    //Calcule des  descripteurs
    extractor.compute( object, kp_object, des_object );

//DEBUT DU PROGRAMME
    int framecount = 0;
	int thresh=100;
    while (1)
    {
        Mat camera;
        cap >> camera;
	
		//initialisation des objets de l'image
        Mat des_image, img_matches,H,image;
        std::vector<KeyPoint> kp_image;
        std::vector<vector<DMatch > > matches;
        std::vector<DMatch > good_matches;
        std::vector<Point2f> obj,scene;
        std::vector<Point2f> scene_corners(4);
    
		//Traitement de l'image camera
        cvtColor(camera, image, CV_RGB2GRAY);
		cv::threshold(image,image,thresh,255,THRESH_BINARY);
		
		//Detection des points Clefs de l'image
        detector.detect( image, kp_image );
		//Calcule des  descripteurs
        extractor.compute( image, kp_image, des_image );

		//Methode: k algorithme-plus proches voisins 
        matcher.knnMatch(des_object, des_image, matches, 2);
        for(int i = 0; i < min(des_image.rows-1,(int) matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
        {	//Application :Premiere methode de Validation croiser
			//on compart l'echantillon  60 % de l echantillon de reference  et sa t aille n'exedent pas 0<echantillon<2 car K=2
            if((matches[i][0].distance < 0.6*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
            {
                good_matches.push_back(matches[i][0]);
            }
        }
        //Trace les lignes de correspondances
        drawMatches( object, kp_object, image, kp_image, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //SI le tableau possede 4 bonne correspondance
		if (good_matches.size() >= 4)
        {	
           for( int i = 0; i < good_matches.size(); i++ )
            {
                //Enregistre les points de Bonne correspondance dans 2 tableau : image Ref et image
                obj.push_back( kp_object[ good_matches[i].queryIdx ].pt );
                scene.push_back( kp_image[ good_matches[i].trainIdx ].pt );
            }
            H = findHomography( obj, scene, CV_RANSAC );
            perspectiveTransform( obj_corners, scene_corners, H);
            //Tracer les 4 lignes de la Transformé
            line( img_matches, scene_corners[0] + Point2f( object.cols, 0), scene_corners[1] + Point2f( object.cols, 0), Scalar(0, 255, 0), 4 );
            line( img_matches, scene_corners[1] + Point2f( object.cols, 0), scene_corners[2] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );
            line( img_matches, scene_corners[2] + Point2f( object.cols, 0), scene_corners[3] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );
            line( img_matches, scene_corners[3] + Point2f( object.cols, 0), scene_corners[0] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );
        }
        //Fenetre de visualisation
		namedWindow("Resulta");
        imshow( "Resulta", img_matches );
		if(cv::waitKey(30) >= 0) break;
    }
	//Destructeur 
	//object.~Mat();
	//des_object.~Mat();
	//kp_object.~vector();
	//obj_corners.~vector();
	destroyAllWindows();
	//cap.release();
}
void Image::FonctionForme()
{
	VideoCapture cap(0);
	Mat image ,image2,rgbCameraFrames,colorTrackingFrames,gray;
	cap>> rgbCameraFrames;
	assert(rgbCameraFrames.type() == CV_8UC3);
	Size size;
	vector<vector<Point> > contours,contours1;
	vector<Mat> planes;
	vector<Vec4i> hier,hier1;
	vector<Vec3f> circles,circles2;
	int thresh=100;

	//Debut du Programme
	for(;;)
	{
		double début;
		début = (double)getTickCount();
		cap>>image;//entrer du flux video dans la matrice image

	//Traitement d image Pour le Cercle Rouge
		image=rgbCameraFrames;
		GaussianBlur(rgbCameraFrames, colorTrackingFrames, Size(11, 11), 0, 0);
		//Isolation de La Gamme de couleur Rouge
		inRange(colorTrackingFrames, Scalar(0, 0, 115), Scalar(50, 50, 255),colorTrackingFrames);
		cv::erode(colorTrackingFrames,colorTrackingFrames,Mat());//on filtre un peu le bruit
		cv::Canny(colorTrackingFrames,colorTrackingFrames,300,100);//les contour de la couleur rouge
		cv::dilate(colorTrackingFrames,colorTrackingFrames,Mat());//on dilate le contour

	//Traitement d image Pour la Fleche 
		cvtColor( image, gray, CV_RGB2GRAY );
		cv::threshold(gray,gray,thresh,255,THRESH_BINARY);
		cv::morphologyEx(gray,gray,CV_MOP_CLOSE,1);
		cv::dilate(gray,gray,Mat());

		//Recherche Des Contours
		findContours(gray,contours,hier,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
		vector <Rect >boundRect(contours.size());
		
		int resultat=0,nbcontour=0;
		vector <Point> pointcont(contours.size());
		
		for(int i=0;i< (int)contours.size();i++)//si il y a des contours Alors :
		{
			double epsilon=3;
			//Approximation du polygone avec plus de précision 
			approxPolyDP( Mat(contours[i]),pointcont,epsilon,true);
			//Calcule de l'Aire 
			resultat =(int)contourArea( pointcont, true );
			//Nombre de Point de la Fleche
			nbcontour=pointcont.size();
			
    //Si l'Aire superieur ou egale a 1000 et le nombre de points egal a 9
			if(resultat >=1000 && nbcontour==9 )
			{
				//Recherche du centre de la fleche
				vector<Moments> mu(2);
				mu[0] = moments( pointcont, false );
				vector<Point2f> mc(2);
				mc[0] = Point2f(mu[0].m10/mu[0].m00 ,mu[0].m01/mu[0].m00 );

				//Tracer La fleche
				drawContours(image,contours,i,Scalar(0,0,255),2,8,vector<Vec4i>(),0,Point());
				rectangle(image,boundRect[i].tl(),boundRect[i].br(),Scalar(0,0,255),2,8,0);

				//Tracer Le centre de la Fleche
				circle( image, mc[0], 4, Scalar(0,255,0), -1, 8, 0 );
				putText(image,"Ma Fleche",mc[0],2,2, Scalar(255,0,0));
				
				//Recherche De cercle 
				HoughCircles( colorTrackingFrames, circles2, CV_HOUGH_GRADIENT, 1,
				colorTrackingFrames.rows/8, 100, 30, 12, 0);

	//Si il y a un ou plusieur cercle 
				if(circles2.size() >=1)
				{
					Point2f center1(cvRound(circles2[0][0]),cvRound(circles2[0][1]));
					int radius1=(int)cvRound(circles2[0][2]);

					//Trace le Cercle Rouge
					putText(image,"cercle rouge",center1,1,2, Scalar(0,255,0));
					circle( image, center1, 3, Scalar(0,255,0), -1, 8, 0 );
					circle( image, center1, radius1, Scalar(0,0,255), 3, 8, 0 );
					cout<<"Centre Fleche X= "<<mc[0].x<<" et Y="<<mc[0].y<<endl;
					cout<<"Centre Cercle I= "<<center1.x<<" et J= "<<center1.y<<endl;
					
					//Trace les lignes qui sont calculer
					line(image, Point(center1.x , center1.y ), Point(mc[0].x , mc[0].y),
					Scalar(0,255,0), 2, CV_AA, 0);
					line(image,Point(mc[0].x,mc[0].y),Point(center1.x,mc[0].y),
					Scalar(255,100,100),4,CV_AA) ;

					//Calcule de l'angle de Rotation
					cout<<" Angle de rotation : "<<PrintAngle1(center1,mc[0])<<endl;
				}
			}
		}
		//Fenetre d'observation
		namedWindow("WEBCAM",1);
		namedWindow("ISOLATION du FONT/PLANT",1);
		namedWindow("objWindow",1);
		imshow("objWindow", colorTrackingFrames);
		imshow("ISOLATION du FONT/PLANT",gray);
		imshow("WEBCAM",image);
		//Sortie de Boucle
		if(cv::waitKey(30) >= 0) break;
		début = ((double)getTickCount() - début)/getTickFrequency();

		cout << "TEMPS en s: " << début << endl << endl;
	}
	//Destructeur
	image.~Mat();
	image2.~Mat();
	rgbCameraFrames.~Mat();
	colorTrackingFrames.~Mat();
	gray.~Mat();
	contours.~vector();
	contours1.~vector();
	planes.~vector();
	circles.~vector();
	circles2.~vector();
	hier.~vector();
	hier1.~vector();
	destroyAllWindows();
	cap.release();
}
void Image:: FonctionSurfModiF()
{
	//INITIALISATION
    Mat object = imread( "fleche_devant1.png", CV_LOAD_IMAGE_GRAYSCALE );
	Mat des_object;
	VideoCapture cap(0);
	std::vector<KeyPoint> kp_object;
    FlannBasedMatcher matcher;
    SurfFeatureDetector detector(500);
	SurfDescriptorExtractor extractor;
	std::vector<Point2f> obj_corners(4);
    //Enregistrement des 4 coins de l'image 
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( object.cols, 0 );
    obj_corners[2] = cvPoint( object.cols, object.rows );
    obj_corners[3] = cvPoint( 0, object.rows );

	vector<vector<Point> > contours;
	vector<Vec4i> hier;

	//Detection des points Clefs de l'image regerence
    detector.detect( object, kp_object );
    //Calcule des  descripteurs
    extractor.compute( object, kp_object, des_object );

//DEBUT DU PROGRAMME
    int framecount = 0;
	int thresh=100;
    while (1)
    {
        Mat camera,Modif,cam;
        cap >> camera;
		cap >> cam;
		assert(cam.type() == CV_8UC3);
		cap>>Modif;
		assert(Modif.type() == CV_8UC3);
		//initialisation des objets de l'image
        Mat des_image, img_matches,H,image;
        std::vector<KeyPoint> kp_image;
        std::vector<vector<DMatch > > matches;
        std::vector<DMatch > good_matches;
        std::vector<Point2f> obj,scene;
        std::vector<Point2f> scene_corners(4);
    
		//Traitement de l'image camera
        cvtColor(camera, image, CV_RGB2GRAY);
		cv::threshold(image,image,thresh,255,THRESH_BINARY);
		
		cvtColor( Modif,Modif, CV_RGB2GRAY );
		cv::threshold(Modif,Modif,thresh,255,THRESH_BINARY);
		cv::morphologyEx(Modif,Modif,CV_MOP_CLOSE,1);
	//	cv::dilate(Modif,Modif,Mat());
		findContours(Modif,contours,hier,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
		vector<vector <Point>> contour_Poly(contours.size());
		vector <Rect >boundRect(contours.size());
		
		//Detection des points Clefs de l'image
        detector.detect( image, kp_image );
		//Calcule des  descripteurs
        extractor.compute( image, kp_image, des_image );

		//Methode: k algorithme-plus proches voisins 
        matcher.knnMatch(des_object, des_image, matches, 2);
        for(int i = 0; i < min(des_image.rows-1,(int) matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
        {	//Application :Premiere methode de Validation croiser
			//on compart l'echantillon  60 % de l echantillon de reference  et sa t aille n'exedent pas 0<echantillon<2 car K=2
            if((matches[i][0].distance < 0.6*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
            {
                good_matches.push_back(matches[i][0]);
            }
        }
        //Trace les lignes de correspondances
        drawMatches( object, kp_object, image, kp_image, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //SI le tableau possede 4 bonne correspondance
		if (good_matches.size() >= 4)
        {
			for(int i=0;i<contours.size();i++)
			{
			
				approxPolyDP( Mat(contours[i]),contour_Poly[i],3,true);
				int resultat =contourArea( contour_Poly[i], true );
				boundRect[i]=boundingRect(Mat(contour_Poly[i]));
				RotatedRect calculeRotation=minAreaRect(contour_Poly[i]);
			
				if(boundRect[i].area()>=1000 && contour_Poly[i].size()==9)
				{
					rectangle(cam,boundRect[i].tl(),boundRect[i].br(),Scalar(0,0,255),2,8,0);
					cout<<" angle de rotation "<<PrintAngle(calculeRotation)<<endl; ;
				}
			}
           for( int i = 0; i < good_matches.size(); i++ )
            {
                //Enregistre les points de Bonne correspondance dans 2 tableau : image Ref et image
                obj.push_back( kp_object[ good_matches[i].queryIdx ].pt );
                scene.push_back( kp_image[ good_matches[i].trainIdx ].pt );
            }
            H = findHomography( obj, scene, CV_RANSAC );
            perspectiveTransform( obj_corners, scene_corners, H);
            //Tracer les 4 lignes de la Transformé
            line( img_matches, scene_corners[0] + Point2f( object.cols, 0), scene_corners[1] + Point2f( object.cols, 0), Scalar(0, 255, 0), 4 );
            line( img_matches, scene_corners[1] + Point2f( object.cols, 0), scene_corners[2] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );
            line( img_matches, scene_corners[2] + Point2f( object.cols, 0), scene_corners[3] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );
            line( img_matches, scene_corners[3] + Point2f( object.cols, 0), scene_corners[0] + Point2f( object.cols, 0), Scalar( 0, 255, 0), 4 );
        }
        //Fenetre de visualisation
		namedWindow("Resulta");
		namedWindow("Res");
        imshow( "Res", cam );
        imshow( "Resulta", img_matches );
		if(cv::waitKey(30) >= 0) break;
    }
	//Destructeur 
	//object.~Mat();
	//des_object.~Mat();
	//kp_object.~vector();
	//obj_corners.~vector();
	destroyAllWindows();
	cap.release();
}

void Image ::FonctionCouleurFleche()
{
	VideoCapture cap(0);//ouverture et initialisation de port capture webcam
	Mat image=imread("fleche_devantN.jpg",1);
	Mat image2=imread("fleche_devantN.jpg",1);
	Mat image3=imread("fleche_devantN.jpg",1);
	Mat stock,Cumul, NCumul;
	Mat stock1,stock2,stock3;
	Mat Res1,Res2;
	vector<vector<Point> > contoursR,contoursV,contoursB;
	vector<Vec4i> hierR,hierB,hierV;
	vector<Vec3f> circles,circles2;//va contenir le cercle
	 
	//couleur rouge
	Mat rouge=image,colorTrackingFramesR; //Matrix 
//	cap>> rouge;
	assert(rouge.type() == CV_8UC3);//on la met sur 8 bit
	//couleur Verte
	Mat vert=image,colorTrackingFramesV; //Matrix 
//	cap>> vert;
	assert(vert.type() == CV_8UC3);//on la met sur 8 bit	
	//couleur Bleu
	Mat bleu=image,colorTrackingFramesB,colorTrackingFramesBL; //Matrix 
//	cap>> bleu;
	assert(bleu.type() == CV_8UC3);//on la met sur 8 bit

	//for(;;)//on fait une boucle infini pour avoir les images en temps Réel
	//{
		//cap>>image;//entrer du flux video dans la matrice image
	
		////////////pour  Rouge ///////////////
		image= rouge;
		GaussianBlur(rouge, colorTrackingFramesR, Size(11, 11), 0, 0); 
		inRange(colorTrackingFramesR, Scalar(0, 0, 115), Scalar(70, 70, 255),colorTrackingFramesR);
		
		////////////pour vert ///////////////
		image2=vert;
		GaussianBlur(vert, colorTrackingFramesV, Size(11, 11), 0, 0); 
		inRange(colorTrackingFramesV, Scalar(20, 50, 20), Scalar(93, 255, 70),colorTrackingFramesV);

		//////////pour Bleu ////////////////
		image3=bleu;
		GaussianBlur(bleu, colorTrackingFramesB, Size(11, 11), 0, 0); 
		inRange(colorTrackingFramesB, Scalar(103, 5, 2), Scalar(252, 78, 73), colorTrackingFramesB);
		cv::dilate(colorTrackingFramesB,colorTrackingFramesB,Mat());
		cv::dilate(colorTrackingFramesB,colorTrackingFramesB,Mat());

		//cercle Vert
		add(colorTrackingFramesB,colorTrackingFramesR,stock1);
		cv::dilate(stock1,stock1,Mat());
		//cercle Rouge 
		add(colorTrackingFramesB,colorTrackingFramesV,stock2);
		cv::dilate(stock2,stock2,Mat());
		//les 2 cercles
		add(colorTrackingFramesV,colorTrackingFramesR,stock3);
		cv::dilate(stock3,stock3,Mat());
		//La fleche 
		add(stock1,stock2,Cumul); 
		
		cv::dilate(Cumul,Cumul,Mat());
		findContours(Cumul,contoursB,hierB,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
		cv::drawContours(image,contoursB,-1,Scalar(0,0,255));

		HoughCircles( stock1, circles,  CV_HOUGH_GRADIENT, 1, 60, 200, 20, 0, 0 );
		HoughCircles( stock2, circles2, CV_HOUGH_GRADIENT, 1, 60, 200, 20, 0, 0 );
		cout<<"le cercle vert  ="<<circles.size()<<endl;
		cout<<"le cercle Rouge  ="<<circles2.size()<<endl;
		if(circles.size() ==1 && circles2.size()==1 )
		{
			Point2f center1(cvRound(circles[0][0]),cvRound(circles[0][1]));
			int radius1=cvRound(circles[0][2]);
			putText(image,"cercle vert",center1,1,2, Scalar(0,0,255));
			circle( image, center1, 3, Scalar(0,0,255), -1, 8, 0 );
			circle( image, center1, radius1, Scalar(0,0,255), 3, 8, 0 );

			Point2f center2(cvRound(circles2[0][0]),cvRound(circles2[0][1]));
			int radius2=cvRound(circles2[0][2]);
			putText(image,"cercle rouge",center2,1,2, Scalar(0,255,0));
			circle( image, center2, 3, Scalar(0,255,0), -1, 8, 0 );
			circle( image, center2, radius2, Scalar(0,255,0), 3, 8, 0 );
		
			cout<<"Centre Fleche X= "<<center2.x<<" et Y="<<center2.y<<endl;
			cout<<"Centre Cercle I= "<<center1.x<<" et J= "<<center1.y<<endl;
			line(image, Point(center2.x , center2.y ), Point(center1.x , center1.y),Scalar(0,255,0), 2, CV_AA, 0);
			line(image,Point(center1.x,center1.y),Point(center2.x,center1.y),Scalar(255,100,100),4,CV_AA) ;
			cout<<" Angle de rotation : "<< PrintAngle1(center2,center1)<<endl ;

		}

		//Fenetre
		namedWindow("WEBCAM",1);
		namedWindow("Cumul",1);
		namedWindow("Bleu et Rouge",1);
		namedWindow("Bleu et Vert",1);
		namedWindow("Vert et Rouge",1);
		namedWindow("Rouge",1);
		namedWindow("Vert",1);
		namedWindow("Bleu",1);

		imshow("Rouge",colorTrackingFramesR);
		imshow("Vert",colorTrackingFramesV);
		imshow("Bleu",colorTrackingFramesB);
		imshow("Cumul",Cumul);
		imshow("Vert et Rouge",stock3);
		imshow("Bleu et Vert",stock2);
		imshow("Bleu et Rouge",stock1);
		imshow("WEBCAM",image);
		waitKey(30);
		system("pause");
		
		//if(cv::waitKey(30) >= 0); //break;
	//}
	//Destructeur
	image.~Mat();
	image2.~Mat();
	image3.~Mat();
	stock.~Mat();
	stock1.~Mat();
	stock2.~Mat();
	stock3.~Mat();
	Cumul.~Mat();
	NCumul.~Mat();
	Res1.~Mat();
	Res2.~Mat();
	contoursB.~vector();
	contoursR.~vector();
	contoursV.~vector();
	hierB.~vector();
	hierR.~vector();
	hierV.~vector();
	circles.~vector();
	circles2.~vector();
	destroyAllWindows();
	cap.release();
}
void Image::FonctionCouleurFlecheCam()
{
	VideoCapture cap(0);//ouverture et initialisation de port capture webcam
	Mat image;//=imread("fleche_devantN.jpg",1);
	Mat image2;//=imread("fleche_devantN.jpg",1);
	Mat image3;//=imread("fleche_devantN.jpg",1);
	Mat stock,Cumul, NCumul;
	Mat stock1,stock2,stock3;
	Mat NR,NB,NV,Res1,Res2;
	Mat NewR,NewV,NewT;
	vector<vector<Point> > contoursR,contoursV,contoursB;

	vector<Vec4i> hierR,hierB,hierV;
	vector<Vec3f> circles,circles2;//va contenir le cercle

	//couleur rouge
	Mat rouge,colorTrackingFramesR; //Matrix 
	cap>> rouge;
	
	assert(rouge.type() == CV_8UC3);//on la met sur 8 bit
	//couleur Verte
	Mat vert,colorTrackingFramesV; //Matrix 
	cap>> vert;
	assert(vert.type() == CV_8UC3);//on la met sur 8 bit	
	//couleur Bleu
	Mat bleu,colorTrackingFramesB,colorTrackingFramesBL; //Matrix 
	cap>> bleu;
	assert(bleu.type() == CV_8UC3);//on la met sur 8 bit

	for(;;)//on fait une boucle infini pour avoir les images en temps Réel
	{

		cap>>image;//entrer du flux video dans la matrice image
	
		////////////pour  Rouge ///////////////
		image= rouge;
		GaussianBlur(rouge, colorTrackingFramesR, Size(11, 11), 0, 0); 
		inRange(colorTrackingFramesR, Scalar(0, 0, 115), Scalar(50, 50, 255),colorTrackingFramesR);
		
		////////////pour vert ///////////////
		image2=vert;
		GaussianBlur(vert, colorTrackingFramesV, Size(11, 11), 0, 0); 
		inRange(colorTrackingFramesV,Scalar(20, 100, 20), Scalar(50, 255, 50),colorTrackingFramesV);

		//////////pour Bleu ////////////////
		image3=bleu;
		GaussianBlur(bleu, colorTrackingFramesB, Size(11, 11), 0, 0); 
		inRange(colorTrackingFramesB, Scalar(100, 20, 20), Scalar(255, 100, 100), colorTrackingFramesB);
		cv::dilate(colorTrackingFramesB,colorTrackingFramesB,Mat());
		cv::dilate(colorTrackingFramesB,colorTrackingFramesB,Mat());

		//cercle Vert
		add(colorTrackingFramesB,colorTrackingFramesR,stock1);//cercle vert dans la fleche
		cv::dilate(stock1,stock1,Mat());
		//cercle Rouge 
		add(colorTrackingFramesB,colorTrackingFramesV,stock2);//cercle rouge dans la fleche
		cv::dilate(stock2,stock2,Mat());
		//les 2 cercles
		add(colorTrackingFramesV,colorTrackingFramesR,stock3);//les 2 cercle dans l image 
		cv::dilate(stock3,stock3,Mat());
		//La fleche 
		add(stock1,stock2,Cumul);//fleche complette 
		

		cv::dilate(Cumul,Cumul,Mat());//fleche complette 
		findContours(Cumul,contoursB,hierB,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
		cv::drawContours(image,contoursB,-1,Scalar(0,0,255));

		HoughCircles( stock1, circles,  CV_HOUGH_GRADIENT, 1, 60, 200, 20, 0, 0 );
		HoughCircles( stock2, circles2, CV_HOUGH_GRADIENT, 1, 60, 200, 20, 0, 0 );
		cout<<"le cercle vert  ="<<circles.size()<<endl;
		cout<<"le cercle Rouge  ="<<circles2.size()<<endl;
		if(circles.size() ==1 && circles2.size()==1)
		{
			Point2f center1(cvRound(circles[0][0]),cvRound(circles[0][1]));
			int radius1=cvRound(circles[0][2]);
			putText(image,"cercle vert",center1,1,2, Scalar(0,0,255));
			circle( image, center1, 3, Scalar(0,0,255), -1, 8, 0 );
			circle( image, center1, radius1, Scalar(0,0,255), 3, 8, 0 );

			Point2f center2(cvRound(circles2[0][0]),cvRound(circles2[0][1]));
			int radius2=cvRound(circles2[0][2]);
			putText(image,"cercle rouge",center2,1,2, Scalar(0,255,0));
			circle( image, center2, 3, Scalar(0,255,0), -1, 8, 0 );
			circle( image, center2, radius2, Scalar(0,255,0), 3, 8, 0 );
		
			cout<<"Centre Fleche X= "<<center2.x<<" et Y="<<center2.y<<endl;
			cout<<"Centre Cercle I= "<<center1.x<<" et J= "<<center1.y<<endl;
			line(image, Point(center2.x , center2.y ), Point(center1.x , center1.y),Scalar(0,255,0), 2, CV_AA, 0);
			line(image,Point(center1.x,center1.y),Point(center2.x,center1.y),Scalar(255,100,100),4,CV_AA) ;
			cout<<" Angle de rotation : "<< PrintAngle1(center2,center1)<<endl ;
		}
		//regarder
		namedWindow("WEBCAM",1);
		namedWindow("Rouge",1);
		namedWindow("Vert",1);
		namedWindow("Bleu",1);

		imshow("Rouge",colorTrackingFramesR);
		imshow("Vert",colorTrackingFramesV);
		imshow("Bleu",colorTrackingFramesB);
		imshow("WEBCAM",image);
		if(cv::waitKey(30) >= 0) break;//pour sortir de la boucle a la pression d une touche
	}
	destroyAllWindows();
	cap.release();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////ETUDE CORRELATION//////////////////////////////////////////////////////////////////
//fonction de creation d image FTMP
IplImage CreationImageFTMP(IplImage *src,IplImage *templ)//src=image camera ,templ=image de reference (une des fleches)
{
//définition de la taille(largeur, hauteur) de l'image ftmp
    int iwidth = src->width - templ->width + 1;
    int iheight = src->height - templ->height + 1;

	//Creer un pointeur d'image ftmp de type IplImage et de taille iwidth et iheight
    IplImage *ftmp = cvCreateImage(cvSize(iwidth,iheight),IPL_DEPTH_32F,1);
	return *ftmp;
}

IplImage CreationImageFTMP_2(IplImage *src,IplImage *templ_2)//src=image camera ,templ=image de reference (une des fleches)
{
//définition de la taille(largeur, hauteur) de l'image ftmp
    int iwidth = src->width - templ_2->width + 1;
    int iheight = src->height - templ_2->height + 1;

	//Creer un pointeur d'image ftmp de type IplImage et de taille iwidth et iheight
    IplImage *ftmp_2 = cvCreateImage(cvSize(iwidth,iheight),IPL_DEPTH_32F,1);
	return *ftmp_2;
}

IplImage CreationImageFTMP_3(IplImage *src,IplImage *templ_3)//src=image camera ,templ=image de reference (une des fleches)
{
//définition de la taille(largeur, hauteur) de l'image ftmp
    int iwidth = src->width - templ_3->width + 1;
    int iheight = src->height - templ_3->height + 1;

	//Creer un pointeur d'image ftmp de type IplImage et de taille iwidth et iheight
    IplImage *ftmp_3 = cvCreateImage(cvSize(iwidth,iheight),IPL_DEPTH_32F,1);
	return *ftmp_3;
}

IplImage CreationImageFTMP_4(IplImage *src,IplImage *templ_4)//src=image camera ,templ=image de reference (une des fleches)
{
//définition de la taille(largeur, hauteur) de l'image ftmp
    int iwidth = src->width - templ_4->width + 1;
    int iheight = src->height - templ_4->height + 1;

	//Creer un pointeur d'image ftmp de type IplImage et de taille iwidth et iheight
    IplImage *ftmp_4 = cvCreateImage(cvSize(iwidth,iheight),IPL_DEPTH_32F,1);
	return *ftmp_4;
}

//Creation du cadre de détection
CvPoint cadre_pt0(IplImage *src,IplImage *templ)
{
    CvPoint cadre_pt1 = cvPoint( (src->width - templ->width) / 2 , (src->height - templ->height) / 2);
	return cadre_pt1;
}

CvPoint cadre_ptbis(IplImage *src,IplImage *templ,CvPoint cadre_pt1)
{
    CvPoint cadre_pt2 = cvPoint(cadre_pt1.x + templ->width , cadre_pt1.y + templ->height);
	return cadre_pt2;
}

//destruction de toutes les fenetres 
void DESTRUCTEUR(CvCapture *capture)
{
	// DESTRUCTEUR 
    cvDestroyAllWindows();
    cvReleaseCapture(&capture);

	return ;
}
void Image::FonctionCorrelation()
{
	//Initilisation d'une touche de clavier
	int key = 1;
	//Flux de la webcam rangée dans capture
    CvCapture* capture = cvCreateCameraCapture( CV_CAP_ANY );
	
	//Définition de 3 images 
    IplImage *src=cvCreateImage(cvSize(640,480), 8, 3);		//image saisi par la camera 
	
	//Chargement des 4 images
	IplImage *templ   = cvLoadImage("fleche_devant.png",1);
	IplImage *templ_2 = cvLoadImage("fleche_reculer.png",1);
	IplImage *templ_3 = cvLoadImage("fleche_droite.png",1);
	IplImage *templ_4 = cvLoadImage("fleche_gauche.png",1);

	//Creation de 4 images qui contiendra l'image resultat de chaque correlation
    IplImage ftmp   = CreationImageFTMP(src,templ);			//image result
	IplImage ftmp_2 = CreationImageFTMP(src,templ_2);
	IplImage ftmp_3 = CreationImageFTMP(src,templ_3);
	IplImage ftmp_4 = CreationImageFTMP(src,templ_4);

	//Creation du cradre qui va définir la zone de détection
    CvPoint cadre_pt1 = cadre_pt0(src,templ);
    CvPoint cadre_pt2 = cadre_ptbis(src,templ,cadre_pt1);

	CvPoint cadre_pt1_2 = cadre_pt0(src,templ_2);
    CvPoint cadre_pt2_2 = cadre_ptbis(src,templ_2,cadre_pt1_2);

	CvPoint cadre_pt1_3 = cadre_pt0(src,templ_3);
    CvPoint cadre_pt2_3 = cadre_ptbis(src,templ_3,cadre_pt1_3);

	CvPoint cadre_pt1_4 = cadre_pt0(src,templ_4);
    CvPoint cadre_pt2_4 = cadre_ptbis(src,templ_4,cadre_pt1_4);

	//Boucle infinie
    while (1)// si il y a flux video 
    {

		//Enregistre les images du flux video dans src
        src = cvRetrieveFrame( capture );

        //applique le filtre médian pour réduire le bruit
        cvSmooth(src,src,CV_MEDIAN,3);
      
//---------------------------CORRELATION---------------------------------------//
		
		//Correlation entre l'image source et l'image de référence. Rangement du résultat dans ftmp
        cvMatchTemplate( src, templ, &ftmp, CV_TM_CCOEFF_NORMED);

		cvMatchTemplate(src, templ_2, &ftmp_2, CV_TM_CCOEFF_NORMED);
		
		cvMatchTemplate(src, templ_3, &ftmp_3, CV_TM_CCOEFF_NORMED);
		
		cvMatchTemplate(src, templ_4, &ftmp_4, CV_TM_CCOEFF_NORMED);
		

        //retrouver dans 'ftmp' les coordonnées du point ayant une valeur maximale
        double min_val , max_val;
		double min_val_2,max_val_2;
		double min_val_3,max_val_3;
		double min_val_4,max_val_4;

        CvPoint min_loc, max_loc;
		CvPoint min_loc_2, max_loc_2;
		CvPoint min_loc_3, max_loc_3;
		CvPoint min_loc_4, max_loc_4;

		//Fonctions qui permettent de calculer le taux maximale de correlation après le passage de la fonction cvMatchTemplate
        cvMinMaxLoc(&ftmp, &min_val, &max_val, &min_loc, &max_loc);

		cvMinMaxLoc(&ftmp_2, &min_val_2, &max_val_2, &min_loc_2, &max_loc_2);

		cvMinMaxLoc(&ftmp_3, &min_val_3, &max_val_3, &min_loc_3, &max_loc_3);

		cvMinMaxLoc(&ftmp_4, &min_val_4, &max_val_4, &min_loc_4, &max_loc_4);

        //défnir un deuxième point à partir du premier point et de la taille de 'ftmp'
        CvPoint max_loc2 = cvPoint(max_loc.x + templ->width, max_loc.y + templ->height);//définir le deuxième point en fonction 
																						//de la taille du template
		CvPoint max_loc2_2 = cvPoint(max_loc_2.x + templ_2->width, max_loc_2.y + templ_2->height);

		CvPoint max_loc2_3 = cvPoint(max_loc_3.x + templ_3->width, max_loc_3.y + templ_3->height);

		CvPoint max_loc2_4 = cvPoint(max_loc_4.x + templ_4->width, max_loc_4.y + templ_4->height);

		//Creation des variables qui contiendra le taux de correlation de chaque image
		int max_val_100 = 0, max_val_100_2 = 0, max_val_100_3 = 0,max_val_100_4 = 0;

		//max_val compris entre [0;1] donc en %, il faut *100
		max_val_100   = (int)(max_val   * 100);
		max_val_100_2 = (int)(max_val_2 * 100);
		max_val_100_3 = (int)(max_val_3 * 100);
		max_val_100_4 = (int)(max_val_4 * 100);

		//Creation du tableau qui va contenir les 4 valeurs de correlation
		int tab[4]={max_val_100,max_val_100_2,max_val_100_3,max_val_100_4};
	
		//Initialisation de 3 variables
		int max=0; //Va contenir le plus grand élément du tableau
		
		//Boucle qui permet de connaitre la plus grande valeur dans le tableau
			for (int i=0 ; i < 4; i++)
			{
				if (tab[i]>max) 
				{
					max=tab[i];
				}		
			}
   
//----------------------Affichage de la direction de la fleche et du taux de correlation------------------//

		if((max == max_val_100) && (max_val_100 > SEUIL))
		{
			cout << "=== FLECHE AVANCER \x85 "<< max_val_100 <<"===" << endl;
			cvNamedWindow( "FLECHE", CV_WINDOW_AUTOSIZE );
			cvShowImage( "FLECHE", templ );
		}

		else if((max == max_val_100_2) && (max_val_100_2 > SEUIL))
		{
			cout << "=== FLECHE DROITE \x85 " << max_val_100_2<< "==="  << endl;
			cvNamedWindow( "FLECHE", CV_WINDOW_AUTOSIZE );
			cvShowImage( "FLECHE", templ_2 );
		}

		else if((max == max_val_100_3) && (max_val_100_3 > SEUIL))
		{	
			cout << "=== FLECHE GAUCHE \x85 " << max_val_100_3<< "==="  << endl;
			cvNamedWindow( "FLECHE", CV_WINDOW_AUTOSIZE );
			cvShowImage( "FLECHE", templ_3);
		}

		else if((max == max_val_100_4) && (max_val_100_4 > SEUIL))
		{	
			cout << "=== FLECHE RECULER \x85 " << max_val_100_4 << "===" << endl;
			cvNamedWindow( "FLECHE", CV_WINDOW_AUTOSIZE );
			cvShowImage( "FLECHE", templ_4);
		}

		else
		{
			cout << "\t=== AUCUN ===" << endl;
			cvDestroyWindow("FLECHE");
		}
		/*
			cout << "------------------------------" << endl;
			cout << "Valeur correlation FDV: " << max_val_100 << endl;
			cout << "Valeur correlation FDR  " << max_val_100_2 << endl << endl;
			cout << "Valeur correlation FD " << max_val_100_3 << endl;
			cout << "Valeur correlation FG " << max_val_100_4 << endl << endl;
	*/
        //si la valeur maximale de 'ftmp' est supérieure au 'seuil'
        //dessiner un rectangle rouge utilisant les coordonnées des deux points 'max_loc' et 'max_loc2'
        if( max_val_100 > SEUIL && max_val!=1 ) 
		{
			cvRectangle(src, max_loc,max_loc2, cvScalar(0,0,255));
		}
		else if( max_val_100_2 > SEUIL && max_val!=1 ) 
		{
			cvRectangle(src, max_loc_2,max_loc2_2, cvScalar(0,0,255));
		}
		else if( max_val_100_3 > SEUIL && max_val!=1 ) 
		{
			cvRectangle(src, max_loc_3,max_loc2_3, cvScalar(0,0,255));
		}
		else if( max_val_100_4 > SEUIL && max_val!=1 ) 
		{
			cvRectangle(src, max_loc_4,max_loc2_4, cvScalar(0,0,255));
		}
		cvNamedWindow( "out", CV_WINDOW_AUTOSIZE );
        cvShowImage( "out", src );
		
		if(cvWaitKey(30) >= 0) break;
    }

   DESTRUCTEUR(capture);
   destroyAllWindows();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////DETECTION PAR LA COULEUR ////////////////////////////////////////////////////////////////////////
//Qu'elle que donner pour identifier plus rapidement certaine couleur 
//Fleche verte :	Hmax : 93	|Hmin : 74	|Smax : 256	|Smin : 25	|Vmax : 109	|Vmin : 132
//Fleche noire :    Hmax : 256  |Smax : 256 |Vmax : 52
//Fleche rouge :    Hmax : 244	|Smax : 256	|Smin : 82 |Vmax : 256 |Vmin : 119
//Valeurs du trackbar initialiser par defaut au demarage 
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
const int MAX_NUM_OBJECTS=50;
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;
//Nom des differentes fenetres
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

void on_trackbar( int, void* )
{//cette fonction est appeler quand la position du trackbar is changer 
}
string intToString(int number)
{// nous renvoi la valeur du nombre 
	std::stringstream ss;
	ss << number;
	return ss.str();
}
void createTrackbars(){
	//creation de la fenetre de track bar
    namedWindow(trackbarWindowName,0);
	//creation de memoir de stockage des parametres
	char TrackbarName[50];
	sprintf( TrackbarName, "H_MIN", H_MIN);
	sprintf( TrackbarName, "H_MAX", H_MAX);
	sprintf( TrackbarName, "S_MIN", S_MIN);
	sprintf( TrackbarName, "S_MAX", S_MAX);
	sprintf( TrackbarName, "V_MIN", V_MIN);
	sprintf( TrackbarName, "V_MAX", V_MAX);
	//creation du trackbar suivant les parametres     
    createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar );
    createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar );
    createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar );
    createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar );
    createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar );
    createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar );
}
void drawObject(int x, int y,Mat &frame){

	//Traver une croix de cible sur l objet tracker 
	circle(frame,Point(x,y),20,Scalar(0,255,0),2);
    if(y-25>0)
    line(frame,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,0),Scalar(0,255,0),2);
    if(y+25<FRAME_HEIGHT)
    line(frame,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,FRAME_HEIGHT),Scalar(0,255,0),2);
    if(x-25>0)
    line(frame,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(0,y),Scalar(0,255,0),2);
    if(x+25<FRAME_WIDTH)
    line(frame,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(FRAME_WIDTH,y),Scalar(0,255,0),2);
	putText(frame,intToString(x)+","+intToString(y),Point(x,y+30),1,1,Scalar(0,255,0),2);

}
void morphOps(Mat &thresh){

	//L'image va etre traiter filtrer pour meilleur rendement
	Mat erodeElement = getStructuringElement( MORPH_RECT,Size(3,3));
	Mat dilateElement = getStructuringElement( MORPH_RECT,Size(8,8));
	erode(thresh,thresh,erodeElement);
	erode(thresh,thresh,erodeElement);
	dilate(thresh,thresh,dilateElement);
	dilate(thresh,thresh,dilateElement);
}
void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed){

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0)
	{
		int numObjects = hierarchy.size();
        //if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
        if(numObjects<MAX_NUM_OBJECTS)
		{
			for (int index = 0; index >= 0; index = hierarchy[index][0]) 
			{

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
                if(area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea){
					x = moment.m10/area;
					y = moment.m01/area;
					objectFound = true;
					refArea = area;
				}else objectFound = false;
			}
			//let user know you found an object
			if(objectFound ==true)
			{
				putText(cameraFeed,"Tracking Object",Point(0,50),2,1,Scalar(0,255,0),2);
				//draw object location on screen
				drawObject(x,y,cameraFeed);
			}

		}else putText(cameraFeed,"TOO MUCH NOISE! ADJUST FILTER",Point(0,50),1,2,Scalar(0,0,255),2);
	}
}

void Image:: FonctionCouleur()
{
    bool trackObjects = true;
    bool useMorphOps = true;
	Mat cameraFeed,HSV,threshold;
	int x=0, y=0;
	//creation du n trackbars HSV 
	createTrackbars();
	VideoCapture capture;
	//ouvrir la capture de l image sur le port 0 c est a dire la camera
	capture.open(0);
	//regler la hauteur et largeur de la camera 
	capture.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);
	cout<<endl;
	cout<<"Qu'elle que donner pour identifier plus rapidement certaine couleur "<<endl<<endl;
	cout<<"Fleche verte : Hmax:93  |Hmin:74 |Smax:256 |Smin:25 |Vmax:109 |Vmin:132"<<endl;
	cout<<"Fleche rouge : Hmax:244 |        |Smax:256 |Smin:82 |Vmax:256 |Vmin:119"<<endl;
	cout<<"Fleche noire : Hmax:256 |        |Smax:256 |        |Vmax:52  |        "<<endl;
	
	//debut de la boucle infini 
	while(1)
	{
		//Enregistre l image capturer dans cette matrice 
		capture.read(cameraFeed);

		//conversion de BGR à HSV colorimétrique
		cvtColor(cameraFeed,HSV,COLOR_BGR2HSV);

		// filtre d'image HSV entre les valeurs et l'image  filtré 
		inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),threshold);

		//effectuer des opérations morphologiques sur l'image seuillée pour éliminer le bruit et mettre l'accent sur l'objet filtré
		if(useMorphOps) morphOps(threshold);

		// passage dans le cadre seuil de notre fonction de suivi d'objet cette fonction retourne les coordonnées x et y 
		if(trackObjects) trackFilteredObject(x,y,threshold,cameraFeed);  

		//ouverture des fenetres  
		imshow(windowName2,threshold);
		imshow(windowName,cameraFeed);
		imshow(windowName1,HSV);
		
		// retard de 30 ms de sorte que l'écran peut rafraîchir. 
		// image n'apparaîtra pas sans cette commande () waitKey
		waitKey(15);
		if(cv::waitKey(30) >= 0) break;
	}
	destroyAllWindows();
}
