#ifndef FONCTION_H
#define FONCTION_H

#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
#include "opencv2\core\core.hpp"

#include "iostream"
#include <fstream>

#include "Windows.h"

using namespace std;
using namespace cv;

#define SEUIL 65

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

//fonction d'ouverture et création  de fenetre
void fonctionOuvertureFenetre(IplImage *src, IplImage *templ_2)
{
	//ouverture et creation des fenetre 
        cvNamedWindow( "out", CV_WINDOW_AUTOSIZE );
        cvShowImage( "out", src );

        /*cvNamedWindow( "template", CV_WINDOW_AUTOSIZE );
        cvShowImage( "template", templ );

		cvNamedWindow( "template_2", CV_WINDOW_AUTOSIZE );
        cvShowImage( "template_2", templ_2 );
		*/
}

//destruction de toutes les fenetres 
void DESTRUCTEUR(CvCapture *capture)
{
	// DESTRUCTEUR 
    cvDestroyAllWindows();
    cvReleaseCapture(&capture);

	return ;
}

//Fonction altitude
int altitude()
{
	//Initialisation de la variable valeur
	int valeur = 0;

	//Lecture du fichier ALTITUDE.txt
	ifstream fichier("ALTITUDE.txt", ios::in);
	
	//Si le fichier existe
	if(fichier)
	{
		//Creation de la variable data qui contiendra la valeur
		//de l'altitude
		int data;
		
		//On rempli la variable data par le contenu du fichier
		fichier >> data;

		//Condition selon la valeur de l'altitude
		if(data<=500)
			valeur = 100;
		else if((500<data)&&(data<1100))
			valeur = 50;
		else if((1100<data)&&(data<1600))
			valeur = 25;
		else if((1600<data)&&(data<2200))
			valeur = 13;
		else
			valeur = 6;

	}
	//On retourne la valeur pour qu'elle soit lu
	return valeur;
}


#endif
