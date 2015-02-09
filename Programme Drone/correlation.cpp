#include "Fonction.h"

//Fonction principale
int main() 
{
	//Initilisation d'une touche de clavier
	int key = 1;
	//Flux de la webcam rangée dans capture
    //CvCapture* capture = cvCreateCameraCapture( CV_CAP_ANY );
	
	//Définition de 3 images 
    IplImage *src=cvCreateImage(cvSize(640,480), 8, 3);		//image saisi par la camera 

	//Chargement des 4 images
	IplImage *haut   = cvLoadImage("fleche_devant.png",1);
	IplImage *droite = cvLoadImage("fleche_droite.png",1);
	IplImage *gauche = cvLoadImage("fleche_gauche.png",1);
	
	//--------------------------------------SUR ECHANTILLONNAGE---------------------------------------//
	
	//==================1ère méthode==============//
	//Varible qui va contenir la pourcentage du redimensionnement 
	//de la nouvelle image en fonciton de l'altitude.
	int percent = 0;

	//Appelle de la fonction altitude
	percent = altitude();

	//Creation d'une nouvelle image qui sera redimensionnée en fonction de l'altitude du drone
	//Cette fonction prends comme argument : 
	// - une nouvelle taille (largeur et hauteur)
	// - une profondeur
	// - un canal
	//On garde la profondeur et le canal de l'image originale, mais on change (cvSize) la taille de l'image
	IplImage *templ = cvCreateImage(
						cvSize((int)((haut->width*percent)/100),(int)((haut->height*percent)/100)),
						haut->depth, 
						haut->nChannels );

	IplImage *templ_2 = cvCreateImage(
						cvSize((int)((droite->width*percent)/100),(int)((droite->height*percent)/100)),
						droite->depth, 
						droite->nChannels );
	IplImage *templ_3 = cvCreateImage(
						cvSize((int)((gauche->width*percent)/100),(int)((gauche->height*percent)/100)),
						gauche->depth, 
						gauche->nChannels );
	
	//On utilise ensuite cvResize pour redéfinir notre nouvelle iamge
	//Argument : 
	// - image source
	// - image template
	cvResize(haut, templ);
	cvResize(droite, templ_2);
	cvResize(gauche, templ_3);
	
	//---------------------------------CORRELATION------------------------------------------//
	
	//Boucle infinie
    while(1)
	{
		
		double début;
		début = (double)getTickCount();

		//Enregistre les images du flux video dans src
		//src = cvRetrieveFrame( capture );
		
		FILE *file = NULL;

		file = fopen("C:/ProjetInformatique/Projet en cours/Autre/Image/images/work.png","r");

		if(file != NULL)
		{
		IplImage *src = cvLoadImage("C:/ProjetInformatique/Projet en cours/Autre/Image/images/work.png",1);
		
        //applique le filtre médian pour réduire le bruit
        cvSmooth(src,src,CV_MEDIAN,3);
		

//---------------------------CORRELATION---------------------------------------//
		
		//Creation de 4 images qui contiendra l'image resultat de chaque correlation
		IplImage ftmp   = CreationImageFTMP(src,templ);
		IplImage ftmp_2 = CreationImageFTMP(src,templ_2);
		IplImage ftmp_3 = CreationImageFTMP(src,templ_3);

		//Creation du cradre qui va définir la zone de détection
		CvPoint cadre_pt1 = cadre_pt0(src,templ);
		CvPoint cadre_pt2 = cadre_ptbis(src,templ,cadre_pt1);

		CvPoint cadre_pt1_2 = cadre_pt0(src,templ_2);
		CvPoint cadre_pt2_2 = cadre_ptbis(src,templ_2,cadre_pt1_2);

		CvPoint cadre_pt1_3 = cadre_pt0(src,templ_3);
		CvPoint cadre_pt2_3 = cadre_ptbis(src,templ_3,cadre_pt1_3);

		//Correlation entre l'image source et l'image de référence. Rangement du résultat dans ftmp
        cvMatchTemplate(src, templ, &ftmp, CV_TM_CCOEFF_NORMED);

		cvMatchTemplate(src, templ_2, &ftmp_2, CV_TM_CCOEFF_NORMED);
		
		cvMatchTemplate(src, templ_3, &ftmp_3, CV_TM_CCOEFF_NORMED);
		
        //retrouver dans 'ftmp' les coordonnées du point ayant une valeur maximale
        double min_val , max_val;
		double min_val_2,max_val_2;
		double min_val_3,max_val_3;

        CvPoint min_loc, max_loc;
		CvPoint min_loc_2, max_loc_2;
		CvPoint min_loc_3, max_loc_3;

		//Fonctions qui permettent de calculer le taux maximale de correlation après le passage de la fonction cvMatchTemplate
        cvMinMaxLoc(&ftmp, &min_val, &max_val, &min_loc, &max_loc);

		cvMinMaxLoc(&ftmp_2, &min_val_2, &max_val_2, &min_loc_2, &max_loc_2);

		cvMinMaxLoc(&ftmp_3, &min_val_3, &max_val_3, &min_loc_3, &max_loc_3);

        //défnir un deuxième point à partir du premier point et de la taille de 'ftmp'
		CvPoint max_loc2 = cvPoint(max_loc.x + templ->width, max_loc.y + templ->height);//définir le deuxième point en fonction 
																						//de la taille du template
		CvPoint max_loc2_2 = cvPoint(max_loc_2.x + templ_2->width, max_loc_2.y + templ_2->height);

		CvPoint max_loc2_3 = cvPoint(max_loc_3.x + templ_3->width, max_loc_3.y + templ_3->height);

		//Creation des variables qui contiendra le taux de correlation de chaque image
		int max_val_100 = 0, max_val_100_2 = 0, max_val_100_3 = 0;

		//max_val compris entre [0;1] donc en %, il faut *100
		max_val_100   = (int)(max_val   * 100);
		max_val_100_2 = (int)(max_val_2 * 100);
		max_val_100_3 = (int)(max_val_3 * 100);

		//Creation du tableau qui va contenir les 4 valeurs de correlation
		int tab[4]={max_val_100,max_val_100_2,max_val_100_3};
	
		//Initialisation de 3 variables
		int max=0; //Va contenir le plus grand élément du tableau
		
		//Boucle qui permet de connaitre la plus grande valeur dans le tableau
			for (int i=0 ; i < 3; i++)
			{
				if (tab[i]>max) 
				{
					max=tab[i];
				}		
			}
   
//----------------------Affichage de la direction de la fleche et du taux de correlation------------------//

ofstream f("C:/ProjetInformatique/Projet en cours/Autre/Image/drone/direction.txt");

		if((max == max_val_100) && (max_val_100 > SEUIL))
		{
			
			int a = 1;
			f << a;

			f.close();

			cout << "=== FLECHE AVANCER \x85 "<< max_val_100 <<"===" << endl;
			cout << "etat : " << a << endl;
			cvNamedWindow( "FLECHE", CV_WINDOW_AUTOSIZE );
			cvShowImage( "FLECHE", haut );
		}

		else if((max == max_val_100_2) && (max_val_100_2 > SEUIL))
		{
			int d = 2;
			f << d;

			f.close();

			cout << "=== FLECHE DROITE \x85 " << max_val_100_2<< "==="  << endl;
			cout << "etat : " << d << endl;
			cvNamedWindow( "FLECHE", CV_WINDOW_AUTOSIZE );
			cvShowImage( "FLECHE", droite );
		}

		else if((max == max_val_100_3) && (max_val_100_3 > SEUIL))
		{	
			int g = 3;
			f << g;

			f.close();

			cout << "=== FLECHE GAUCHE \x85 " << max_val_100_3<< "==="  << endl;
			cout << "etat : " << g << endl;
			cvNamedWindow( "FLECHE", CV_WINDOW_AUTOSIZE );
			cvShowImage( "FLECHE", gauche);
		}

		else
		{
			cout << "\t=== AUCUN ===" << endl;
			cvDestroyWindow("FLECHE");
		}
		
			//cout << "=== FLECHE AVANCER \x85 "<< max_val_100 <<"===" << endl;
			/*cout << "=== FLECHE DROITE \x85 " << max_val_100_2<< "==="  << endl;
			cout << "=== FLECHE GAUCHE \x85 " << max_val_100_3<< "==="  << endl;
			cout << "=== FLECHE RECULER \x85 " << max_val_100_4 << "===\n" << endl;*/
       
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
		

		fonctionOuvertureFenetre(src,templ_2);

		//On attends
		key= cvWaitKey(10);

		début = ((double)getTickCount() - début)/getTickFrequency();
		cout << "TEMPS : " << début << endl << endl;
		remove("C:/ProjetInformatique/Projet en cours/Autre/Image/images/work.png");
		}
		Sleep(1000);
		remove("C:/ProjetInformatique/Projet en cours/Autre/Image/images/work.png");
    }
   //DESTRUCTEUR(capture);
   return 0;
}

