// Format:  face_detect2 step_size merge_radius
//			cross_thresh classifier_name

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <io.h>

#define H_PAD 4
#define W_PAD 4

#define HSIZE 24
#define WSIZE 24

char debug = 0;
int total_detect=0;
int processed = 0;



// This is the same feature_struct that's in face_train.c
// This is the same feature_struct that's in face_train.c
struct feature_struct {
    char type;			// Type of feature
    char left;			// Left boundary of feature
    char top;			// Top boundary of feature
    char right;			// Right boundary of feature
    char bottom;		// Bottom boundary of feature
    int thresh;
    char pos_neg;
    double error;		// Error of the feature relative to the image in question
    double linear_error;
    int pos_center,neg_center;
    int num;
};

// Basic image_struct, only capable of having rgb pictures as opposed to just grayscale.
struct image_struct {
	double **image;
	unsigned char **r;
	unsigned char **g;
	unsigned char **b;
	double **int_image;
	int height;
	int width;
	int color;
	double weight;
};

// The classifer.
struct classifier_struct {
	int num_features;
	struct feature_struct *features;
	double thresh;
};

// Struct of detected images.  When a face is found, it's thrown into this struct.
struct detected_struct {
	int cent_x;
	int cent_y;
	int left;
	int right;
	int top;
	int bottom;
	int weight;
};

// Various bitmap structs necessary for reading and writing .bmp files.
struct BMPHeader {
	int bfSize;
	short bfReserved1;
	short bfReserved2;
	unsigned int bfOffbits;
};

struct BMPInfoHeader {
	int biSize;
	int biWidth;
	int biHeight;
	short biPlanes;
	short biBitCount;
	int biCompression;
	int biSizeImage;
	int biXPelsPerMeter;
	int biYPelsPerMeter;
	int biClrUsed;
	int biClrImportant;
};

struct BMPPalette {
	char r;
	char g;
	char b;
	char reserved;
};
// End bitmap structs.

struct classifier_struct *classifiers;
int    height, width;

void draw_feature(struct image_struct* image, struct feature_struct* feature);

struct image_struct load_image(char *filename, int *height, int *width);
void integral_image(double **input, double **output, int width, int height);
int apply_feature(double **image, struct feature_struct* feature, int x, int y, double scale);
void write_image(struct image_struct image, char *filename, int type);
int apply_thresh_feature(double **image, struct feature_struct feature, int x, int y, double scale);
void unload(struct image_struct old);
int apply_thresh_class(struct classifier_struct classifier, double **image,int x, int y, double scale,double& score);
unsigned char **extract_image(unsigned char **image, int x, int y, int height, int width);
void clean_image(struct image_struct* img);
char* evaluate_features(struct classifier_struct* cls);
void pad_image(image_struct& image);
void img_mean_var(image_struct* img,double& mean, double& sigma);
void img_mean_var_correction(image_struct* img,double mean, double sigma);

void alloc_image(int w, int h, image_struct* temp_image);
void alloc_int_image(int w, int h, image_struct* temp_image);
void move_img(image_struct& img,image_struct& shift_img,int dx, int dy);
void crop_img(image_struct& img, image_struct& cropped, int x0,int y0,int dx, int dy);

void draw_box(struct image_struct* image,int t,int l, int b,int r,char color);
void resample_bilinear(image_struct& img, image_struct& reshape,double ratio);
void add_spacing(image_struct& img,int dh,int dw);
int get_num_features(image_struct& img);

classifier_struct* read_text_classifiers(char* filename,int& height,int& width);


int main(int argc, char **argv) {
	//temp variables
	int    x;
	//double scale;
	char   filename[80];
    char*   path;
	int    test_set;
	FILE  *output;
	struct image_struct temp_image,crop_image,scale_image,out_image;

	struct detected_struct *centers;
	int detected = 0;
 
	//main control loop variables
	int clas_loop;

    char single_img = 0;
    double score;

    char* response_vector = NULL;
    char* format_str = "";
    int add_img;

    double start_scale = 0.8;
    double end_scale = 1.2;

    int num_imgs = 2000;
    int eat_whole_directory = 1;

	int mask_height = HSIZE,mask_width=WSIZE;
    

    if(argc<3)
        {
        printf("Usage: %s -p|-n|-i|-pp|-nn clas_file [img file]\n",argv[0]);
        exit(1);
        }
 
    if(!stricmp(argv[1],"-p"))
        {
        if(argc!=3)
            {
            printf("Usage: %s -p clas_file\n",argv[0]);
            exit(1);
            }
        format_str = "../positive/positive%04d.pgm";
        add_img = 2000;
        eat_whole_directory = 0;
        }
    else if(!stricmp(argv[1],"-pp"))
        {
        if(argc!=3)
        {
            printf("Usage: %s -pp clas_file\n",argv[0]);
            exit(1);
        }
        format_str = "../positive/positive%04d.pgm";
        add_img = 1;
        eat_whole_directory = 0;
        }
    else if(!stricmp(argv[1],"-pa"))
    {
        if(argc!=3)
        {
            printf("Usage: %s -pa clas_file\n",argv[0]);
            exit(1);
        }
        format_str = "../48x48/andy_pos/scface%05d.pgm";
        add_img = 0;
        start_scale = 1.6;
        end_scale = 2.8;
        num_imgs = 2429;
        eat_whole_directory = 0;
    }
    else if(!stricmp(argv[1],"-na"))
    {
        if(argc!=3)
        {
            printf("Usage: %s -na clas_file\n",argv[0]);
            exit(1);
        }
        format_str = "../48x48/andy_neg/*.pgm";
        path = "../48x48/andy_neg/";
        add_img = 0;
        start_scale = 1.8;
        end_scale = 2.6;
        num_imgs = 5000;
    }
    else if(!stricmp(argv[1],"-pbf"))
    {
        if(argc!=3)
        {
            printf("Usage: %s -pbf clas_file\n",argv[0]);
            exit(1);
        }
        format_str = "../48x48/baris_front/*.pgm";
        path = "../48x48/baris_front/";
        add_img = 0;
        start_scale = 1.8;
        end_scale = 2.6;
        num_imgs = 5000;
    }
    else if(!stricmp(argv[1],"-pbm"))
    {
        if(argc!=3)
        {
            printf("Usage: %s -pbm clas_file\n",argv[0]);
            exit(1);
        }
        format_str = "../48x48/baris_mid/*.pgm";
        path = "../48x48/baris_mid/";
        add_img = 0;
        start_scale = 1.8;
        end_scale = 2.6;
        num_imgs = 5000;
    }
    else if(!stricmp(argv[1],"-pw"))
    {
        if(argc!=3)
        {
            printf("Usage: %s -pw clas_file\n",argv[0]);
            exit(1);
        }
        format_str = "../48x48/west/*.pgm";
        path = "../48x48/west/";
        add_img = 0;
        start_scale = 1.8;
        end_scale = 2.6;
        num_imgs = 5000;
    }
	else if(!stricmp(argv[1],"-po"))
	{
		if(argc!=3)
		{
			printf("Usage: %s -po clas_file\n",argv[0]);
			exit(1);
		}
		format_str = "../48x48/original/*.pgm";
		path = "../48x48/original/";
		add_img = 0;
		start_scale = 3.0;
		end_scale = 4.6;
		num_imgs = 5000;
	}
	else if(!stricmp(argv[1],"-pc"))
	{
		if(argc!=3)
		{
			printf("Usage: %s -pc clas_file\n",argv[0]);
			exit(1);
		}
		format_str = "../48x48/crop/*.pgm";
		path = "../48x48/crop/";
		add_img = 0;
		start_scale = 2.0;
		end_scale = 6.0;
		num_imgs = 5000;
	}
    else if(!stricmp(argv[1],"-n0"))
    {
        if(argc!=3)
        {
            printf("Usage: %s -n0 clas_file\n",argv[0]);
            exit(1);
        }
        format_str = "../48x48/01-1m/*.pgm";
        path = "../48x48/01-1m/";
        add_img = 0;
        start_scale = 1.8;
        end_scale = 2.6;
        num_imgs = 5000;
    }
    else if(!stricmp(argv[1],"-ns"))
    {
        if(argc!=3)
        {
            printf("Usage: %s -ns clas_file\n",argv[0]);
            exit(1);
        }
        format_str = "../48x48/squirrel_pos/*.pgm";
        path = "../48x48/squirrel_pos/";
        add_img = 0;
        start_scale = 1.8;
        end_scale = 2.6;
        num_imgs = 5000;
    }
    else if(!stricmp(argv[1],"-no"))
    {
        if(argc!=3)
        {
            printf("Usage: %s -no clas_file\n",argv[0]);
            exit(1);
        }
        format_str = "../48x48/original_non/*.pgm";
        path = "../48x48/original_non/";
        add_img = 0;
        start_scale = 1.8;
        end_scale = 2.6;
        num_imgs = 5000;
    }
    else if(!stricmp(argv[1],"-n"))
        {
        if(argc!=3)
            {
            printf("Usage: %s -n clas_file\n",argv[0]);
            exit(2);
            }
        format_str = "../scenery/scenery%05d.pgm";
        add_img = 12000;
        eat_whole_directory = 0;
        }
    else if(!stricmp(argv[1],"-nn"))
    {
        if(argc!=3)
        {
            printf("Usage: %s -nn clas_file\n",argv[0]);
            exit(2);
        }
        format_str = "../scenery/scenery%05d.pgm";
        add_img = 1;
        eat_whole_directory = 0;
    }
    else if(!stricmp(argv[1],"-i"))
        {
        if(argc!=4)
            {
            printf("Usage: %s -i clas_file img_file\n",argv[0]);
            exit(3);
            }
        single_img = 1;
        }

	

	if(strstr(argv[2],".txt"))
		{
		classifiers = read_text_classifiers("new.txt",mask_height,mask_width);
		}
	else
		{
		output=fopen(argv[2],"rb");
		// Begin if.
		if(output!=NULL) 
			{
			fread(&clas_loop,sizeof(int),1,output);
			classifiers = (struct classifier_struct *) malloc((clas_loop+1)*sizeof(struct classifier_struct));
			printf("clas_loop: %d\n",clas_loop);
			fread(classifiers,sizeof(struct classifier_struct),clas_loop+1,output);
			// Begin for.
			for(x = 0;x <= clas_loop;x++) 
				{
				printf("Classifier %d: Thresh: %f Features: %d\n",x,classifiers[x].thresh,classifiers[x].num_features);
				classifiers[x].features = (struct feature_struct *) malloc((classifiers[x].num_features)*sizeof(struct feature_struct));
				fread(classifiers[x].features,sizeof(struct feature_struct),classifiers[x].num_features,output);
				}	// End for.
			fclose(output);
			}		// End if.
		// Begin else.
		else 
			{
			printf("classifier file not found\n");
			return 0;
			}		// End else.
		centers = NULL;
		}


	

    if(single_img==1)
        {
        temp_image=load_image(argv[3], &(height), &(width));
        temp_image.int_image = NULL;
        out_image = load_image(argv[3], &(height), &(width));

        if(temp_image.image == NULL) {
            printf("Error: input image not found\n");
            return 1;
            }

        // loop through all points. toDo- add scaling
        // run (unscaled) detector
        // filter out rectangles that are too close together
        //draw boxes around faces
		double scale = 1.4;
        resample_bilinear(temp_image,scale_image,scale);
		write_image(scale_image,"scale.pgm",0);
        height = scale_image.height;
        width = scale_image.width;

        alloc_image(mask_width,mask_height,&crop_image);
        int num_detected = 0;
		char filename[80];

        for(int x=0;x<height-mask_height/2;x+=1)
        for(int y=0;y<width-mask_width/2;y+=1)
                {
                crop_img(scale_image,crop_image,x,y,mask_height,mask_width);  
				//sprintf(filename,"crop%d-%d.pgm",x,y);
				//write_image(crop_image,filename,0);
                integral_image(crop_image.image,crop_image.int_image,crop_image.width,crop_image.height);    
                int res = apply_thresh_class(classifiers[0],crop_image.int_image,0,0,1,score);
                num_detected+=res;
                if(res) draw_box(&out_image,x*scale,y*scale,(x+mask_height)*scale,(y+mask_width)*scale,100);
                }

        printf("Faces detected: %d \n",num_detected);

        write_image(out_image,"detected.pgm",0);

        unload(temp_image);
        unload(crop_image);

        while(1);
        return 1;
        }


    int first_search = 1;
    long hfile,hfile1;
    
    for(test_set = 1;test_set <= num_imgs;test_set++) {
	// Begin while.   (Or begin for.)
        
		detected = 0;
        struct _finddata_t pgm_file;

        if(!eat_whole_directory)
            {
            sprintf(filename,format_str,test_set+add_img);
            }
		else if(first_search==1)
            {
            hfile = _findfirst(format_str,&pgm_file);
            if(hfile==-1L) return -1;
            first_search = 0;
            sprintf(filename,"%s%s",path,pgm_file.name);
            }    
        else if(first_search==0)
            {
            hfile1=_findnext(hfile,&pgm_file);
            if(hfile1==-1)
                {
                printf("Done here.No more pgm files in the directory\n");
                break;
                }

            sprintf(filename,"%s%s",path,pgm_file.name);
            }

		temp_image=load_image(filename, &(height), &(width));
		
		if(test_set%100==0)
			printf("Image: %s \n",filename);
		height = temp_image.height;
		width = temp_image.width;
		// Begin if.
		if(temp_image.image == NULL) {
			printf("Error: input image not found\n");
			continue;
		    }		// End if.

        double mean, var;
                
        int res=0;
        double scale = 0.9;
        int xpos =0;
        int ypos=0;
        image_struct shift_image;
		int num_features;
        
        for(scale=start_scale;scale<=end_scale;scale+=0.2)
            {
            //scale image 
			
            resample_bilinear(temp_image,scale_image,scale);

/*			
            if(scale>=end_scale-0.2) 
				{
				num_features = get_num_features(scale_image);
				printf("num features %d\n",num_features);
				}
*/
            //unload(temp_image);
            if(scale_image.height<mask_height)
                {
                int dh = mask_height - scale_image.height;
                int dw = mask_width - scale_image.width;
                add_spacing(scale_image,dh,dw);
                }

            alloc_image(scale_image.width,scale_image.height,&shift_image);

            int min_xpos = (scale_image.height-mask_height+1);
            int min_ypos = (scale_image.width-mask_width+1);
            //write_image(scale_image,"scale_test.pgm",0);
            //if image is smaller than 24x24, pad image
            //(xpos,ypos)goes from  from -(24-w+1,24-h+1) to (1,1)
            for(int xpos=-min_xpos;xpos<=2;xpos++)
                for(int ypos=-min_ypos;ypos<=2;ypos++)
                {
                //move image

                move_img(scale_image,shift_image,xpos,ypos);
                //char filename[80];
                //sprintf(filename,"shift%d%d.pgm",xpos,ypos);
                //write_image(shift_image,filename,0);

                //img_mean_var(&shift_image,mean,var);
                //img_mean_var_correction(&shift_image,mean,var);

                integral_image(shift_image.image,shift_image.int_image,shift_image.width,shift_image.height);
                //recompute int_image
                res += apply_thresh_class(classifiers[0],shift_image.int_image,0,0,1,score);                    
                }
            unload(scale_image);
            unload(shift_image);
            } //end scale

        
        unload(temp_image);
                    
		// Begin for (1).
		if(res>0) 
			{
			//printf("Num detections %d \n",res);
			total_detect++;
			}
        processed++;
        }

  
printf("Detected %d out of %d, percentage %5.3f\n",total_detect,processed,(float)100*total_detect/processed);
while(1);
/*
printf("Constructing integral face: integral_face.pgm\n");
integral_face = load_image("../positive/positive0001.pgm",&integral_face.height,&integral_face.width);
clean_image(&integral_face);

response_vector = evaluate_features(classifiers);

for(i=0;i<classifiers->num_features;i++) 
    {
    if(response_vector[i] != 1 && (abs(classifiers->features[i].thresh)<10000))
        draw_feature(&integral_face,&classifiers->features[i]);
    }

write_image(integral_face,"integral_face.pgm",0);

free(response_vector);
*/

return 0;
}

void draw_rect(struct image_struct* image,int t,int l, int b,int r,char color)
{
int x,y;

printf("t=%d l=%d b=%d r=%d\n",t,l,b,r);

for(x=t;x<=b;x++)
    for(y=l;y<=r;y++)
        {
        image->image[x][y]+=color;
        }
}

void draw_box(struct image_struct* image,int t,int l, int b,int r,char color)
{
    int x,y;

    printf("t=%d l=%d b=%d r=%d\n",t,l,b,r);

    
    
    for(x=t;x<b;x++) 
        {
        if(x<0 || x>=image->height) continue;
        if(l>=0 && l<image->width) image->image[x][l]=color;
        if(r>=0 && r<image->width) image->image[x][r]=color;
        }
    
    for(y=l;y<r;y++)
        {
        if(y<0 || y>=image->width) continue;
        if(t>=0 && t<image->height) image->image[t][y]=color;
        if(b>=0 && b<image->height) image->image[b][y]=color;
        }
}


void draw0feature(struct image_struct* image, struct feature_struct* feature,char color)
{
    //AlexK toDo adjust threshold
    //Assuming symmetry for now
    int t1,l1,r1,b1;
    int t2,l2,r2,b2;


    t1=feature->top;
    l1=feature->left;
    r1=feature->right;
    b1=(feature->top+feature->bottom)/2;

    t2=(feature->top+feature->bottom)/2+1;
    l2=feature->left;
    r2=feature->right;
    b2=feature->bottom;

    if(feature->pos_neg == 1)
        {
        draw_rect(image,t1,l1,b1,r1,color);
        draw_rect(image,t2,l2,b2,r2,-color);
        }
    else if(feature->pos_neg == -1)
        {
        draw_rect(image,t1,l1,b1,r1,-color);
        draw_rect(image,t2,l2,b2,r2,color);
        }
    
}

void draw1feature(struct image_struct* image, struct feature_struct* feature,char color)
{
    //AlexK toDo adjust threshold
    //Assuming symmetry for now
    int t1,l1,r1,b1;
    int t2,l2,r2,b2;

    t1=feature->top;
    l1=feature->left;
    r1=(feature->right+feature->left)/2;
    b1=feature->bottom;

    t2=feature->top;
    l2=(feature->left+feature->right)/2+1;
    r2=feature->right;
    b2=feature->bottom;

    if(feature->pos_neg == 1)
        {
        draw_rect(image,t1,l1,b1,r1,color);
        draw_rect(image,t2,l2,b2,r2,-color);
        }
    else if(feature->pos_neg == -1)
        {
        draw_rect(image,t1,l1,b1,r1,-color);
        draw_rect(image,t2,l2,b2,r2,color);
        }
    
}

void draw2feature(struct image_struct* image, struct feature_struct* feature,char color)
{
    //AlexK toDo adjust threshold
    //Assuming symmetry for now
    int t1,l1,r1,b1;
    int t2,l2,r2,b2;
    int t3,l3,r3,b3;
    int t4,l4,r4,b4;

    t1=feature->top;
    l1=feature->left;
    r1=(feature->right+feature->left)/2;
    b1=(feature->bottom+feature->top)/2;

    t2=(feature->bottom+feature->top)/2+1;
    l2=(feature->right+feature->left)/2+1;
    r2=feature->right;
    b2=feature->bottom;

    t3=feature->top;
    l3=(feature->left+feature->right)/2;
    r3=feature->right;
    b3=(feature->top+feature->bottom)/2;

    t4 = (feature->top+feature->bottom)/2+1;
    l4 = feature->left;
    r4 = (feature->left+feature->right)/2-1;
    b4 = feature->bottom;

    printf("feature type 2 \n");
    if(feature->pos_neg == 1)
        {
        printf("positive \n");

        draw_rect(image,t1,l1,b1,r1,color);
        draw_rect(image,t2,l2,b2,r2,color);
        draw_rect(image,t3,l3,b3,r3,-color);
        draw_rect(image,t4,l4,b4,r4,-color);
        
        }
    else if(feature->pos_neg == -1)
        {
        printf("negative \n");
        draw_rect(image,t1,l1,b1,r1,-color);
        draw_rect(image,t2,l2,b2,r2,-color);
        draw_rect(image,t3,l3,b3,r3,color);
        draw_rect(image,t4,l4,b4,r4,color);
        }  

}

void draw3feature(struct image_struct* image, struct feature_struct* feature,char color)
{
    //AlexK toDo adjust threshold
    //Assuming symmetry for now
    int t1,l1,r1,b1;
    int t2,l2,r2,b2;
    int t3,l3,r3,b3;

    t1=feature->top;
    l1=feature->left;
    r1=feature->right;
    b1=(2*feature->top+feature->bottom)/3;

    t3 = (feature->top+2*feature->bottom)/3+1;
    l3 = feature->left;
    r3 = feature->right;
    b3 = feature->bottom;

    t2=(2*feature->top+feature->bottom)/3+1;
    l2=feature->left;
    r2=feature->right;
    b2=(feature->top+2*feature->bottom)/3;

    printf("feature type 3 \n");
    if(feature->pos_neg == 1)
        {
        printf("positive \n");
        draw_rect(image,t1,l1,b1,r1,-color);
        draw_rect(image,t2,l2,b2,r2,3*color);
        draw_rect(image,t3,l3,b3,r3,-color);
        }
    else if(feature->pos_neg == -1)
        {
        printf("negative \n");
        draw_rect(image,t1,l1,b1,r1,color);
        draw_rect(image,t2,l2,b2,r2,-3*color);
        draw_rect(image,t3,l3,b3,r3,color);
        }
}

void draw4feature(struct image_struct* image, struct feature_struct* feature,char color)
{
    //AlexK toDo adjust threshold
    //Assuming symmetry for now
    int t1,l1,r1,b1;
    int t2,l2,r2,b2;
    int t3,l3,r3,b3;

    l2=(2*feature->left+feature->right)/3+1;
    t2=feature->top;
    b2=feature->bottom;
    r2=(feature->left+2*feature->right)/3;

    l1=feature->left;
    t1=feature->top;
    b1=feature->bottom;
    r1=(2*feature->left+feature->right)/3;

    t3 = feature->top;
    l3 = (feature->left+2*feature->right)/3+1;
    r3 = feature->right;
    b3 = feature->bottom;

    printf("feature type 4 \n");
    if(feature->pos_neg == 1)
        {
        printf("positive \n");
        draw_rect(image,t1,l1,b1,r1,-color);
        draw_rect(image,t2,l2,b2,r2,color);
        draw_rect(image,t3,l3,b3,r3,-color);
        }
    else if(feature->pos_neg == -1)
        {
        printf("negative \n");
        draw_rect(image,t1,l1,b1,r1,color);
        draw_rect(image,t2,l2,b2,r2,-color);
        draw_rect(image,t3,l3,b3,r3,color);
        }


}

void draw_feature(struct image_struct* image, struct feature_struct* feature)
{
//calculate alpha value
double alpha;

double coeff = 32.0/log((1.0-0.2)/0.2);

char color;


alpha = log((1-feature->error)/feature->error); //range 0..128 

printf("alpha = %5.3f \n",alpha );

color = (int)(alpha*coeff);

printf("color = %d \n",color);

switch(feature->type)
    {
    case 0:
        draw0feature(image,feature,color);
    break;
    case 1:
        draw1feature(image,feature,color);
    break;
    case 2:
        draw2feature(image,feature,color);
    break;
    case 3:
        draw3feature(image,feature,color);
    break;
    case 4:
        draw4feature(image,feature,color);
    break;
    }

};



//-----------------------------------------------------------------------------------------------------
void integral_image(double **input, double **output, int width, int height) {
	int x;
	int y;
	double current_row_sum;

	for(x = 0;x <= height;x++) {
		output[x][0] = 0;
	}
	for(x = 0;x <= width;x++) {
		output[0][x] = 0;
	}
	for(x = 0;x < height;x++) {
		current_row_sum = 0;
		for(y = 0;y < width;y++) {
			current_row_sum += input[x][y];
			if(x == 0) {
				output[(x + 1)][(y + 1)] = current_row_sum;
			}
			else {
				output[(x + 1)][(y + 1)] = output[x][(y + 1)] + current_row_sum;
			}
		}
	}
}

//-----------------------------------------------------------------------------------------------------
//return 1 if classifier thinks it's a face 0 if it's not a face
int apply_thresh_class(struct classifier_struct classifier, double **image,int x, int y, double scale,double& score) {
	double score1 = 0.0;
	double score2 = 0.0;
	int z;
	static int max_score=-1000;
	//printf("The threshold is %5.3f \n",classifier.thresh);
	classifier.thresh = 0.62;
	for(z = 0;z < classifier.num_features;z++) {
		if(z==0) debug = 1;
		int res = apply_thresh_feature(image, classifier.features[z], x, y,scale);
		//printf("Feature %d thresh %d pos_neg %d error %5.3f thinks: %d \n",z,classifier.features[z].thresh,classifier.features[z].pos_neg,classifier.features[z].error,res);
		score1 += (double) res * log((1 - classifier.features[z].error)/classifier.features[z].error);
		score2 += log((1 - classifier.features[z].error) / classifier.features[z].error);
		debug = 0;
	}
	
	score = score1-classifier.thresh * score2;

	if(score>max_score)
		{
		max_score = (int)score;
		//printf("x=%d y=%d thresh %5.3f score diff %5.3f\n",x,y,classifier.thresh,score);
		}
   
	//printf("x=%d y=%d thresh %5.3f score diff %5.3f\n",x,y,classifier.thresh,score);
	//printf("apply_thresh_class score1 %5.3f score2 %5.3f thresh %5.3f \n",score1,score2,classifier.thresh);

	if(score1 >= classifier.thresh * score2) {		
		return 1;
	}
	else {
		return 0;
	}
}

//-------------------------------------------------------------------------------------------------
//return 1 if is a face and 0 if is not a face
int apply_thresh_feature_old(double **image, struct feature_struct feature, int x, int y, double scale) {
	if(feature.thresh < (feature.pos_neg*apply_feature(image,&feature,x,y,scale))) {
		return 1;
	}
	return 0;
}

int apply_thresh_feature(double **image, struct feature_struct feature, int x, int y, double scale) {

	int feature_value = apply_feature(image, &feature, x, y,scale);
    /*
	if(debug)
		printf("Feature value %d\n",feature_value);
    */
	if (feature.pos_neg == 1) {
		if (feature_value > feature.thresh) {
			return 1;
		}
		else {
			return 0;
		}
	}
	if (feature.pos_neg == -1) {
		if (feature_value < feature.thresh) {
			return 1;
		}
		else {
			return 0;
		}
	}
	else {
		return 0;
	}
}


//-------------------------------------------------------------------------------------------------
//returns the output value of the feature
int apply_feature_old(double **image, struct feature_struct feature, int x, int y, double scale) {
    double holder;
	switch(feature.type) {
	case 0:
		holder =   ((image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*feature.left))+y]
				-image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*feature.right))+y]
				+(2*image[((int) (0.5+scale*((feature.bottom+feature.top)/2)))+x][((int) (0.5+scale*feature.right))+y])
				-(2*image[((int) (0.5+scale*((feature.bottom+feature.top)/2)))+x][((int) (0.5+scale*feature.left))+y])
				+image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*feature.left))+y]
				-image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*feature.right))+y])/*/(scale*scale)*/);
		break;
	case 1:
		holder =  ((image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*feature.left))+y]
				-(2*image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*((feature.left+feature.right)/2)))+y])
				+image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*feature.right))+y]
				-image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*feature.left))+y]
				+(2*image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*((feature.left+feature.right)/2)))+y])
				-image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*feature.right))+y])/*/(scale*scale)*/);
		break;
	case 2:
		holder =   ((image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*feature.left))+y]
				-(2*image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*((feature.left+feature.right)/2)))+y])
				+image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*feature.right))+y]
				-(2*image[((int) (0.5+scale*((feature.top+feature.bottom)/2)))+x][((int) (0.5+scale*feature.left))+y])
				+(4*image[((int) (0.5+scale*((feature.top+feature.bottom)/2)))+x][((int) (0.5+scale*((feature.left+feature.right)/2)))+y])
				-(2*image[((int) (0.5+scale*((feature.top+feature.bottom)/2)))+x][((int) (0.5+scale*feature.right))+y])
				+image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*feature.left))+y]
				-(2*image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*((feature.left+feature.right)/2)))+y])
				+image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*feature.right))+y])/*/(scale*scale)*/);
		break;
	case 3:
		holder = ((image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*feature.right))+y]
				-image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*feature.left))+y]
				+(4*image[((int) (0.5+scale*((2*feature.top+feature.bottom)/3)))+x][((int) (0.5+scale*feature.left))+y])
				-(4*image[((int) (0.5+scale*((2*feature.top+feature.bottom)/3)))+x][((int) (0.5+scale*feature.right))+y])
				-(4*image[((int) (0.5+scale*((feature.top+2*feature.bottom)/3)))+x][((int) (0.5+scale*feature.left))+y])
				+(4*image[((int) (0.5+scale*((feature.top+2*feature.bottom)/3)))+x][((int) (0.5+scale*feature.right))+y])
				+image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*feature.left))+y]
				-image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*feature.right))+y])/*/(scale*scale)*/);
	case 4:
		holder = ((image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*feature.right))+y]
				-image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*feature.left))+y]
				+(4*image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*((2*feature.left+feature.right)/3)))+y])
				-(4*image[((int) (0.5+scale*feature.top))+x][((int) (0.5+scale*((feature.left+2*feature.right)/3)))+y])
				-(4*image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*((2*feature.left+feature.right)/3)))+y])
				+(4*image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*((feature.left+2*feature.right)/3)))+y])
				+image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*feature.left))+y]
				-image[((int) (0.5+scale*feature.bottom))+x][((int) (0.5+scale*feature.right))+y])/*/(scale*scale)*/);
		break;
	case 5:
		holder = ((image[((int) (scale*feature.top))+x][((int) (scale*((3*feature.left+feature.right)/4)))+y]
				-image[((int) (scale*feature.top))+x][((int) (scale*((feature.left+3*feature.right)/4)))+y]
				-image[((int) (scale*((3*feature.top+2*feature.bottom)/5)))+x][((int) (scale*((3*feature.left+feature.right)/4)))+y]
				+image[((int) (scale*((3*feature.top+2*feature.bottom)/5)))+x][((int) (scale*((feature.left+3*feature.right)/4)))+y]
				+image[((int) (scale*((3*feature.top+2*feature.bottom)/5)))+x][((int) (scale*feature.left))+y]
				-image[((int) (scale*((3*feature.top+2*feature.bottom)/5)))+x][((int) (scale*((feature.left+feature.right)/2)))+y]
				-image[((int) (scale*((feature.top+4*feature.bottom)/5)))+x][((int) (scale*feature.left))+y]
				+image[((int) (scale*((feature.top+4*feature.bottom)/5)))+x][((int) (scale*((feature.left+feature.right)/2)))+y]
				-image[((int) (scale*((4*feature.top+feature.bottom)/5)))+x][((int) (scale*((feature.left+feature.right)/2)))+y]
				+image[((int) (scale*((4*feature.top+feature.bottom)/5)))+x][((int) (scale*feature.right))+y]
				+image[((int) (scale*((2*feature.top+3*feature.bottom)/5)))+x][((int) (scale*((feature.left+feature.right)/2)))+y]
				-image[((int) (scale*((2*feature.top+3*feature.bottom)/5)))+x][((int) (scale*feature.right))+y]
				-image[((int) (scale*((2*feature.top+3*feature.bottom)/5)))+x][((int) (scale*((3*feature.left+feature.right)/4)))+y]                       
				+image[((int) (scale*((2*feature.top+3*feature.bottom)/5)))+x][((int) (scale*((feature.left+3*feature.right)/4)))+y]                       
				+image[((int) (scale*feature.bottom))+x][((int) (scale*((3*feature.left+feature.right)/4)))+y]
				-image[((int) (scale*feature.bottom))+x][((int) (scale*((feature.left+3*feature.right)/4)))+y])/(scale*scale));
		break;
	case 6:
		holder = ((image[((int) (scale*feature.top))+x][((int) (scale*((4*feature.left+feature.right)/5)))+y]
				-image[((int) (scale*feature.top))+x][((int) (scale*((2*feature.left+3*feature.right)/5)))+y]
				-image[((int) (scale*((feature.top+feature.bottom)/2)))+x][((int) (scale*((4*feature.left+feature.right)/5)))+y]
				+image[((int) (scale*((feature.top+feature.bottom)/2)))+x][((int) (scale*((2*feature.left+3*feature.right)/5)))+y]
				+image[((int) (scale*((3*feature.top+feature.bottom)/4)))+x][((int) (scale*((2*feature.left+3*feature.right)/5)))+y]
				-image[((int) (scale*((3*feature.top+feature.bottom)/4)))+x][((int) (scale*feature.right))+y]
				-image[((int) (scale*((feature.top+3*feature.bottom)/4)))+x][((int) (scale*((2*feature.left+3*feature.right)/5)))+y]
				+image[((int) (scale*((feature.top+3*feature.bottom)/4)))+x][((int) (scale*feature.right))+y]
				-image[((int) (scale*((3*feature.top+feature.bottom)/4)))+x][((int) (scale*feature.left))+y]
				+image[((int) (scale*((3*feature.top+feature.bottom)/4)))+x][((int) (scale*((3*feature.left+2*feature.right)/5)))+y]
				+image[((int) (scale*((feature.top+3*feature.bottom)/4)))+x][((int) (scale*feature.left))+y]                      
				-image[((int) (scale*((feature.top+3*feature.bottom)/4)))+x][((int) (scale*((3*feature.left+2*feature.right)/5)))+y]
				-image[((int) (scale*((feature.top+feature.bottom)/2)))+x][((int) (scale*((3*feature.left+2*feature.right)/5)))+y]
				+image[((int) (scale*((feature.top+feature.bottom)/2)))+x][((int) (scale*((feature.left+4*feature.right)/5)))+y]
				+image[((int) (scale*feature.bottom))+x][((int) (scale*((3*feature.left+2*feature.right)/5)))+y]                      
				-image[((int) (scale*feature.bottom))+x][((int) (scale*((feature.left+4*feature.right)/5)))+y])/(scale*scale));
		break;
	default:
		return 0;
	}

    int res = int(holder);
    return res;
}

//----------------------------------------------------------------------------------------------------------------------
//returns the output value of the feature
int apply_feature(double**image, struct feature_struct* feature, int x, int y,double scale) {
    double holder;
    int i1,i2,i3,i4,i5,i6;
    switch(feature->type) {
        case 0:
            //printf("Case 0\n");
            //printf("top: %d right: %d left: %d bottom: %d\n", feature->top, feature->right, feature->left, feature->bottom);
            i1 = (int)(feature->bottom)+x;
            i2 = (int)(feature->left)+y;
            i3 = (int)(feature->right)+y;
            i4 = (int) ((feature->bottom+feature->top)/2)+x;
            i5 = (int) (feature->top)+x;

            holder =  ((image[i1][i2]
            -image[i1][i3]
            +(2*image[i4][i3])
                -(2*image[i4][i2])
                +image[i5][i2]
                -image[i5][i3]));
                //printf("Holder: %d\n", holder);
                break;
        case 1:
            //printf("Case 1\n");
            //printf("top: %d right: %d left: %d bottom: %d\n", feature->top, feature->right, feature->left, feature->bottom);
            i1 = ((int) (feature->top))+x;
            i2 = ((int) (feature->left))+y;
            i3 = ((int) (((feature->left+feature->right)/2)))+y;
            i4 = ((int) (feature->right))+y;
            i5 = ((int) (feature->bottom))+x;
            holder =  ((image[i1][i2]
            -(2*image[i1][i3])
                +image[i1][i4]
                -image[i5][i2]
                +(2*image[i5][i3])
                    -image[i5][i4]));
                //printf("Holder: %d\n", holder);
                break;
        case 2:
            //printf("Case 2\n");
            //printf("top: %d right: %d left: %d bottom: %d\n", feature->top, feature->right, feature->left, feature->bottom);
            i1 = ((int) (feature->top))+x;
            i2 = ((int) (feature->left))+y;
            i3 = ((int) (((feature->left+feature->right)/2)))+y;
            i4 = ((int) (feature->right))+y;
            i5 = ((int) (((feature->top+feature->bottom)/2)))+x;
            i6 = ((int) (feature->bottom))+x;


            holder =   ((image[i1][i2]
            -(2*image[i1][i3])
                +image[i1][i4]
                -(2*image[i5][i2])
                    +(4*image[i5][i3])
                    -(2*image[i5][i4])
                    +image[i6][i2]
                    -(2*image[i6][i3])
                        +image[i6][i4]));
                    //printf("Holder: %d\n", holder);
                    break;
        case 3:
            //printf("Case 3\n");
            //printf("top: %d right: %d left: %d bottom: %d\n", feature->top, feature->right, feature->left, feature->bottom);
            i1 = ((int) (feature->top))+x;
            i2 = ((int) (feature->right))+y;
            i3 = ((int) (feature->left))+y;
            i4 = ((int) (((2*feature->top+feature->bottom)/3)))+x;
            i5 = ((int) (((feature->top+2*feature->bottom)/3)))+x;
            i6 = ((int) (feature->bottom))+x;
            holder =   ((image[i1][i2]
            -image[i1][i3]
            +(4*image[i4][i3])
                -(4*image[i4][i2])
                -(4*image[i5][i3])
                +(4*image[i5][i2])
                +image[i6][i3]
                -image[i6][i2]));
                //printf("Holder: %d\n", holder);
                break;
        case 4:
            //printf("Case 4\n");
            //printf("top: %d bottom: %d left: %d right: %d\n", feature->top, feature->bottom, feature->left, feature->right);
            i1 = ((int) (feature->top))+x;
            i2 = ((int) (feature->right))+y;
            i3 = ((int) (feature->left))+y;
            i4 = ((int) (((2*feature->left+feature->right)/3)))+y;
            i5 = ((int) (feature->bottom))+x;
            i6 = ((int) (((feature->left+2*feature->right)/3)))+y;
            holder =   ((image[i1][i2]
            -image[i1][i3]
            +(4*image[i1][i4])
                -(4*image[i1][i6])
                -(4*image[i5][i4])
                +(4*image[i5][i6])
                +image[i5][i3]
                -image[i5][i2]));
                //printf("Holder: %d\n", holder);
                break;
        case 5:
            //printf("Case 5\n");
            //printf("top: %d right: %d left: %d bottom: %d\n", feature->top, feature->right, feature->left, feature->bottom);
            holder =  ((image[((int) (feature->top))+x][((int) (((3*feature->left+feature->right)/4)))+y]
            -image[((int) (feature->top))+x][((int) (((feature->left+3*feature->right)/4)))+y]
            -image[((int) (((3*feature->top+2*feature->bottom)/5)))+x][((int) (((3*feature->left+feature->right)/4)))+y]
            +image[((int) (((3*feature->top+2*feature->bottom)/5)))+x][((int) (((feature->left+3*feature->right)/4)))+y]
            +image[((int) (((3*feature->top+2*feature->bottom)/5)))+x][((int) (feature->left))+y]
            -image[((int) (((3*feature->top+2*feature->bottom)/5)))+x][((int) (((feature->left+feature->right)/2)))+y]
            -image[((int) (((feature->top+4*feature->bottom)/5)))+x][((int) (feature->left))+y]
            +image[((int) (((feature->top+4*feature->bottom)/5)))+x][((int) (((feature->left+feature->right)/2)))+y]
            -image[((int) (((4*feature->top+feature->bottom)/5)))+x][((int) (((feature->left+feature->right)/2)))+y]
            +image[((int) (((4*feature->top+feature->bottom)/5)))+x][((int) (feature->right))+y]
            +image[((int) (((2*feature->top+3*feature->bottom)/5)))+x][((int) (((feature->left+feature->right)/2)))+y]
            -image[((int) (((2*feature->top+3*feature->bottom)/5)))+x][((int) (feature->right))+y]
            -image[((int) (((2*feature->top+3*feature->bottom)/5)))+x][((int) (((3*feature->left+feature->right)/4)))+y]
            +image[((int) (((2*feature->top+3*feature->bottom)/5)))+x][((int) (((feature->left+3*feature->right)/4)))+y]
            +image[((int) (feature->bottom))+x][((int) (((3*feature->left+feature->right)/4)))+y]
            -image[((int) (feature->bottom))+x][((int) (((feature->left+3*feature->right)/4)))+y]));
            //printf("Holder: %d\n", holder);
            break;
        case 6:
            //printf("Case 6\n");
            //printf("top: %d right: %d left: %d bottom: %d\n", feature->top, feature->right, feature->left, feature->bottom);
            holder =   ((image[((int) (feature->top))+x][((int) (((4*feature->left+feature->right)/5)))+y]
            -image[((int) (feature->top))+x][((int) (((2*feature->left+3*feature->right)/5)))+y]
            -image[((int) (((feature->top+feature->bottom)/2)))+x][((int) (((4*feature->left+feature->right)/5)))+y]
            +image[((int) (((feature->top+feature->bottom)/2)))+x][((int) (((2*feature->left+3*feature->right)/5)))+y]
            +image[((int) (((3*feature->top+feature->bottom)/4)))+x][((int) (((2*feature->left+3*feature->right)/5)))+y]
            -image[((int) (((3*feature->top+feature->bottom)/4)))+x][((int) (feature->right))+y]
            -image[((int) (((feature->top+3*feature->bottom)/4)))+x][((int) (((2*feature->left+3*feature->right)/5)))+y]
            +image[((int) (((feature->top+3*feature->bottom)/4)))+x][((int) (feature->right))+y]
            -image[((int) (((3*feature->top+feature->bottom)/4)))+x][((int) (feature->left))+y]
            +image[((int) (((3*feature->top+feature->bottom)/4)))+x][((int) (((3*feature->left+2*feature->right)/5)))+y]
            +image[((int) (((feature->top+3*feature->bottom)/4)))+x][((int) (feature->left))+y]
            -image[((int) (((feature->top+3*feature->bottom)/4)))+x][((int) (((3*feature->left+2*feature->right)/5)))+y]
            -image[((int) (((feature->top+feature->bottom)/2)))+x][((int) (((3*feature->left+2*feature->right)/5)))+y]
            +image[((int) (((feature->top+feature->bottom)/2)))+x][((int) (((feature->left+4*feature->right)/5)))+y]
            +image[((int) (feature->bottom))+x][((int) (((3*feature->left+2*feature->right)/5)))+y]
            -image[((int) (feature->bottom))+x][((int) (((feature->left+4*feature->right)/5)))+y]));
            //printf("Holder: %d\n", holder);
            break;
        default:
            return 0;
    }

    int res = int(holder);
    return res;
}

void clean_image(struct image_struct* img)
{
int x,y;
for(x=0;x<img->height;x++)
    for(y=0;y<img->width;y++)
        {
        img->image[x][y]=127;
        }
}

void pad_image(image_struct& image)
{
int h_shift =H_PAD/2;
int w_shift =W_PAD/2;
int i,j,k;
for(k=image.height-H_PAD;k>=1;k--)
    {
    for(i=0;i<k;i++) image.image[i+h_shift][k-1+w_shift]=image.image[i][k-1];
    for(j=0;j<k;j++) image.image[k-1+h_shift][j+w_shift]=image.image[k-1][j];
    }


for(i=0;i<image.height;i++)
    for(j=0;j<image.width;j++)
        {
        if(i>(image.height-h_shift-1))
            image.image[i][j]=image.image[i-1][j]*0.1;
        if(j>(image.width-w_shift-1))
            image.image[i][j]=image.image[i][j-1]*0.1;
        }

        
}


struct image_struct load_image(char *filename, int *height, int *width) {
	FILE *input;
	char c;
	char comment[50];
	int x, y;
	int r,g,b;
	struct image_struct temp;

	temp.color=0;
	input = fopen(filename,"rb");
	if(input == NULL) {
		temp.image = NULL;
		return temp;
	}
	c = getc(input);
	if(c=='B') {
		struct BMPHeader Head;
		struct BMPInfoHeader InfHead;
		int bytes_in_row;
  
		c = getc(input);
		fread(&Head,sizeof(struct BMPHeader),1,input);
		fread(&InfHead,sizeof(struct BMPInfoHeader),1,input);

		if(InfHead.biBitCount == 24) {
			temp.color = 1;
			temp.height = InfHead.biHeight;
			temp.width = InfHead.biWidth;
			temp.image = (double **) malloc(temp.height * sizeof(double *));
			temp.r = (unsigned char **) malloc(temp.height * sizeof(unsigned char *));
			temp.g = (unsigned char **) malloc(temp.height * sizeof(unsigned char *));
			temp.b = (unsigned char **) malloc(temp.height * sizeof(unsigned char *));
			for(x = (temp.height - 1);x >= 0;x--) {
				temp.image[x] = (double *) malloc(temp.width * sizeof(double));
				temp.r[x] = (unsigned char *) malloc(temp.width * sizeof(unsigned char));
				temp.g[x] = (unsigned char *) malloc(temp.width * sizeof(unsigned char));
				temp.b[x] = (unsigned char *) malloc(temp.width * sizeof(unsigned char));
				bytes_in_row = 0;
   				for(y = 0;y < temp.width;y++) {
   					temp.b[x][y] = getc(input);
   					temp.g[x][y] = getc(input);
   					temp.r[x][y] = getc(input);
   					temp.image[x][y] = (222 * temp.r[x][y] + 707 * temp.g[x][y] + 71 * temp.b[x][y]) / 1000;
   					bytes_in_row += 3;
   				}
   				while((bytes_in_row % 4) != 0) {
   					getc(input);
   					bytes_in_row++;
   				}
			}
		}
		else {
			temp.image=NULL;
		}
		fclose(input);
		return temp;
	}
	else if(c == 'P') {
		c = getc(input);
		if(c == '5') {
			fscanf(input," ");
			c = fgetc(input);
			while(c == '#') {
				fgets(comment,40,input);
				while(comment[strlen(comment)-1]!='\n') {
					fgets(comment,40,input);
				}
			fscanf(input," ");
			c = fgetc(input);
			}
			ungetc(c,input);
			fscanf(input," %d %d ",&(temp.width), &(temp.height));
			c = fgetc(input);
			while(c == '#') {
				fgets(comment,40,input);
				while(comment[strlen(comment)-1]!='\n') {
					fgets(comment,40,input);
				}
			fscanf(input," ");
			c = fgetc(input);
			}
			ungetc(c,input);
			fscanf(input," %d",&x);
			fgetc(input);

            //temp.height+=H_PAD;
            //temp.width+=W_PAD;
            temp.int_image = NULL;

			temp.image = (double **) malloc(temp.height * sizeof(double *));

			for(x = 0;x < (temp.height);x++) {
				temp.image[x] = (double *) malloc((temp.width) * sizeof(double));
                memset(temp.image[x],0,temp.width*sizeof(double));
				for(y = 0;y < (temp.width);y++) {
					temp.image[x][y] = fgetc(input);
				}
			}
			fclose(input);
            *height = temp.height;
            *width = temp.width;
			return temp;
		}
		else if(c=='6') {
			temp.color = 1;
			fscanf(input," ");
			c = fgetc(input);
			while(c == '#') {
				fgets(comment,40,input);
				while(comment[strlen(comment)-1]!='\n') {
				    fgets(comment,40,input);
				}
				fscanf(input," ");
				c = fgetc(input);
			}
			ungetc(c,input);
			fscanf(input," %d %d ",&(temp.width), &(temp.height));
			c = fgetc(input);
			while(c == '#') {
				fgets(comment,40,input);
				while(comment[strlen(comment)-1]!='\n') {
					fgets(comment,40,input);
				}
				fscanf(input," ");
				c = fgetc(input);
			}
			ungetc(c,input);
			fscanf(input," %d",&x);
			fgetc(input);
			temp.image = (double **) malloc(temp.height * sizeof(double *));
			temp.r = (unsigned char **) malloc(temp.height * sizeof(unsigned char *));
			temp.g = (unsigned char **) malloc(temp.height * sizeof(unsigned char *));
			temp.b = (unsigned char **) malloc(temp.height * sizeof(unsigned char *));

			for(x = 0;x < temp.height;x++) {
				temp.image[x] = (double *) malloc(temp.width * sizeof(double));
				temp.r[x] = (unsigned char *) malloc(temp.width * sizeof(unsigned char));
				temp.g[x] = (unsigned char *) malloc(temp.width * sizeof(unsigned char));
				temp.b[x] = (unsigned char *) malloc(temp.width * sizeof(unsigned char));
				for(y = 0;y < temp.width;y++) {
					r = fgetc(input);
					g = fgetc(input);
					b = fgetc(input);
					temp.image[x][y] = (222 * r + 707 * g + 71 * b) / 1000;
					temp.r[x][y] = r;
				    temp.g[x][y] = g;
					temp.b[x][y] = b;
				}
			}
			fclose(input);
			return temp;
		}
		else {
			fclose(input);
			temp.image = NULL;
			return temp;
		}
	}
	else {
		fclose(input);
		temp.image = NULL;
		return temp;
	}
}

//----------------------------------------------------------------------------------------------
void unload(struct image_struct old) {
	int x;
	for(x = 0;x<old.height;x++) {
        
		free(old.image[x]);
        if(old.int_image!=NULL)
		    free(old.int_image[x]);
        old.image[x]=NULL;
        if(old.int_image!=NULL)
            old.int_image[x]=NULL;
		if(old.color == 1) {
			free(old.r[x]);
			free(old.g[x]);
			free(old.b[x]);
		}
	}

    if(old.int_image!=NULL)
        {
	    free(old.int_image[x]);
        old.int_image[x]=NULL;
        }
    
	free(old.image);
    old.image = NULL;
	if(old.color == 1) {
		free(old.r);
		free(old.g);
		free(old.b);
	}
    if(old.int_image!=NULL)
	    free(old.int_image);
    old.int_image = NULL;
}

//---------------------------------------------------------------------------------------------
unsigned char **extract_image(unsigned char **image, int x, int y, int height, int width) {
	unsigned char **image2;
	int x1;
	int y1;
	image2 = (unsigned char **) malloc(height * sizeof(unsigned char *));
	for(x1 = 0;x1 < height;x1++) {
		image2[x1] = (unsigned char *) malloc(width * sizeof(unsigned char));
		for(y1 = 0;y1 < width;y1++) {
			image2[x1][y1] = image[(x + x1)][(y + y1)];
		}
	}
	return image2;
}

//---------------------------------------------------------------------------------------------
void write_image(struct image_struct image, char *filename, int type) {
	FILE *output;
	int x,y;
	output = fopen(filename,"wb");
	switch(type) {
	case 0:
		fprintf(output,"P5\n%d %d\n255\n",image.width, image.height);
		for(x = 0;x < image.height;x++) {
			for(y = 0;y < image.width;y++) {
				fprintf(output,"%c",unsigned char(image.image[x][y]));
			}
		}
		break;
	case 1:
		fprintf(output,"P6\n%d %d\n255\n",image.width, image.height);
		if(image.color == 1) {
			for(x = 0;x < image.height;x++) {
				for(y = 0;y < image.width;y++) {
					fprintf(output,"%c%c%c",image.r[x][y],image.g[x][y],image.b[x][y]);
				}
			}
		}
		else {
			for(x = 0;x < image.height;x++) {
				for(y = 0;y < image.width;y++) {
					fprintf(output,"%c%c%c",image.image[x][y],image.image[x][y],image.image[x][y]);
				}
			}
		}
        break;
    case 2: {
        struct BMPHeader Head;
        struct BMPInfoHeader InfHead;
        int bytes_in_row;
        fprintf(output,"BM");
        Head.bfOffbits=54;
        Head.bfReserved1 = 0;
        Head.bfReserved2 = 0;
        Head.bfSize = 54 + ((image.width * 3 - 1) / 4 + 1) * 4;
        InfHead.biSize = 40;
        InfHead.biWidth = image.width;
        InfHead.biHeight = image.height;
        InfHead.biPlanes = 1;
        InfHead.biBitCount = 24;
        InfHead.biCompression = 0;
        InfHead.biSizeImage = 0;
        InfHead.biXPelsPerMeter = 7500;
        InfHead.biYPelsPerMeter = 7500;
        InfHead.biClrUsed = 0;
        InfHead.biClrImportant = 0;
        fwrite(&Head,sizeof(struct BMPHeader),1,output);
        fwrite(&InfHead,sizeof(struct BMPInfoHeader),1,output);
        if(image.color == 1) {
            for(x = (image.height - 1);x >= 0;x--) {
                bytes_in_row=0;
                for(y = 0;y < image.width;y++) {
                    fprintf(output,"%c%c%c",image.b[x][y],image.g[x][y],image.r[x][y]);
                    bytes_in_row += 3;
                }
                while((bytes_in_row % 4) != 0) {
                    fprintf(output,"%c",0);
                    bytes_in_row++;
                }
            }
        }
        else {
            for(x = (image.height - 1);x >= 0;x--) {
                bytes_in_row = 0;
                for(y = 0;y < image.width;y++) {
                    fprintf(output,"%c%c%c",image.image[x][y],image.image[x][y],image.image[x][y]);
                    bytes_in_row += 3;
                }
                while((bytes_in_row % 4) != 0) {
                    fprintf(output,"%c",0);
                    bytes_in_row++;
                }
            }
        }
            }
            break;
    }
    fclose(output);
}

void load_and_calc(struct image_struct* image,char* filename)
{
    int x;
    *image=load_image(filename, &(height), &(width));
    image->int_image = (double **) malloc((image->height+1)*sizeof(double *));
    // Begin for.
    for(x = 0;x <= image->height;x++) {
        image->int_image[x] = (double *) malloc((image->width+1)*sizeof(double));
    }		// End for.

    integral_image(image->image,image->int_image,image->width,image->height);
}

//determine if a feature is likely to be a face or a non-face
char* evaluate_features(struct classifier_struct* cls)
{
    int* face_response =NULL;
    int* nonface_response = NULL;
    char* response_vector = NULL;
    char filename[80];
    struct image_struct temp_image;

    int i;
    int j;

    int size = cls->num_features*sizeof(int);

    char res;

    face_response = (int*)malloc(size);
    nonface_response = (int*)malloc(size);
    response_vector = (char*)malloc(cls->num_features);

    memset(face_response,0,size);
    memset(nonface_response,0,size);



    //evaluate positives
    for(i=0;i<200;i++)
    {
        sprintf(filename,"../positive/positive%04d.pgm",i+1000);
        load_and_calc(&temp_image,filename);
        //evaluate features on the image
        for(j=0;j<cls->num_features;j++)
        {
            res = apply_thresh_feature(temp_image.int_image,cls->features[j],0,0,1);
            if(res) face_response[j]++;
            else nonface_response[j]++;
        }
        unload(temp_image);
    }

    //evaluate negatives
    for(i=0;i<200;i++)
    {
        sprintf(filename,"../scenery/scenery%05d.pgm",i+1000);
        load_and_calc(&temp_image,filename);
        //evaluate features on the image
        for(j=0;j<cls->num_features;j++)
        {
            res = apply_thresh_feature(temp_image.int_image,cls->features[j],0,0,1);
            if(res) nonface_response[j]++;
            else face_response[j]++;
        }
        unload(temp_image);
    }

    //decide if a feature is + or -
    for(j=0;j<cls->num_features;j++)
    {
        printf("feature[%d] face_response %d nonface_response %d\n", j,face_response[j],nonface_response[j] );
        if(face_response[j]>200 && nonface_response[j]<200) 
        {
            response_vector[j] = 1; //face
        }
        else if(face_response[j]<200 && nonface_response[j]>200)
        {
            response_vector[j]= 2;
        }
        else 
        {
            response_vector[j] = 0;
        }
    }

    free(face_response);
    free(nonface_response);

    return response_vector;
}

void img_mean_var(image_struct* img,double& mean, double& sigma)
{
    mean = 0;
    sigma = 0;
    for(int i=0;i<img->width;i++)
        for(int j=0;j<img->height;j++)
        {
            mean+=img->image[i][j];
            sigma+= (img->image[i][j])*(img->image[i][j]);
        }
        mean/=(img->width*img->height);
        sigma/=(img->width*img->height);
        sigma = sigma-mean*mean;
        sigma = sqrt(sigma);
}

void img_mean_var_correction(image_struct* img,double mean, double sigma)
{
    double value;
    for(int i=0;i<img->width;i++)
        for(int j=0;j<img->height;j++)
        {
            value =img->image[i][j];
            value = value-mean;
            value = value/sigma;
            img->image[i][j]=value;
        }

}

void alloc_image(int w, int h, image_struct* temp_image)
{
int x;


temp_image->image = (double **) malloc(h * sizeof(double *));

for(x = 0;x < h;x++) 
    {
    temp_image->image[x] = (double *) malloc(w * sizeof(double));
    //memset(temp_image->image[x],0,w*sizeof(double));
    }

temp_image->int_image = (double **) malloc((h+1)*sizeof(double *));
// allocate integral image rows
for(x = 0;x <= h;x++) temp_image->int_image[x] = (double *) malloc((w+1)*sizeof(double));
temp_image->width = w;
temp_image->height = h;

temp_image->color = 0;

}

void alloc_int_image(int w, int h, image_struct* temp_image)
{
    temp_image->int_image = (double **) malloc((h+1)*sizeof(double *));
    // allocate integral image rows
    for(int x = 0;x <= h;x++) temp_image->int_image[x] = (double *) malloc((w+1)*sizeof(double));
    temp_image->width = w;
    temp_image->height = h;
}

void move_img(image_struct& img,image_struct& shift_img,int dx, int dy)
{

for(int x=0;x<img.height;x++)
    for(int y=0;y<img.width;y++)
        shift_img.image[x][y]=img.image[x][y];

for(int x=0;x<img.height;x++)
    for(int y=0;y<img.width;y++)
        {
        int xnew,ynew;
        xnew = x+dx;
        ynew = y+dy;
        if(xnew<0 ||xnew>=img.height) continue;
        if(ynew<0 ||ynew>=img.width) continue;
        shift_img.image[xnew][ynew]=img.image[x][y];
        }

}

void crop_img(image_struct& img, image_struct& cropped, int x0,int y0,int dx, int dy)
{

//create img (dx,dy)
//alloc_image(dx,dy,&cropped);

//copy pixels (x0,y0,dx,dy)
for(int x=x0;x<x0+dx;x++)
for(int y=y0;y<y0+dy;y++)
    {
    int xnew,ynew;
    xnew = x-x0;
    ynew = y-y0;
    if(x>=img.height) continue;
    if(y>=img.width) continue;
    cropped.image[xnew][ynew]=img.image[x][y];
    }

}

void resample_bilinear(image_struct& img, image_struct& reshape,double ratio)
{
//allocate reshaped
int new_height = ceil((double)img.height/ratio);
int new_width = ceil((double)img.width/ratio);

int h = img.height;
int w = img.width;

double alpha, beta;
int fx,fy;


int inw,ine,isw,ise;

alloc_image(new_width,new_height,&reshape);

char debug = 0;
if(debug)
	{
	unload(reshape);
	}

int ix,iy;
ix = 0;
for(double x=0;x<h;x+=ratio)
    {    
    if(ix>=new_height) continue;
    iy=0;
    fx = floor(x);
    alpha = x-fx;
    for(double y=0;y<w;y+=ratio)
        {        
        if(iy>=new_width) continue;
        fy=floor(y);

        if((fy+1) >= w || (fx+1)>=h)
            {
            reshape.image[ix][iy] = img.image[fx][fy];
            }
        else
            {
            beta = y-fy;
            reshape.image[ix][iy]=(1-alpha)*(1-beta)*img.image[fx][fy] 
            + (1-alpha)*beta*img.image[fx][fy+1] + alpha*(1-beta)*img.image[fx+1][fy] + alpha*beta*img.image[fx+1][fy+1];
            }
        
        iy++;
        }
    ix++;
    }
}

void add_spacing(image_struct& img,int dh,int dw)
{
image_struct temp;
int new_height = img.height + dh;
int new_width = img.width + dw;
alloc_image(new_width,new_height,&temp);
for(int x=0;x<new_height;x++)
    for(int y=0;y<new_width;y++)
        {
        if(x<img.height && y<img.width)
            {
            temp.image[x][y]=img.image[x][y];
            }
        else
            {
            temp.image[x][y]=0;
            }
        }
unload(img);
img = temp;
}

int get_num_features(image_struct& img)
{
int num_features = 0;
for(int x = 1;x<img.height-1;x++)
	for(int y = 1;y<img.width-1;y++)
		{
		double min_value = 255;
		double max_value = 0;
		for(int x1=x-1;x1<x+1;x1++)
			for(int y1=y-1;y1<y+1;y1++)
				{
				if(x1==x && y1==y) continue;
				if(min_value>img.image[x1][y1]) min_value = img.image[x1][y1];
				if(max_value<img.image[x1][y1]) max_value = img.image[x1][y1];
				}
		if(img.image[x][y]<=min_value || img.image[x][y]>=max_value) num_features++;
		}
return num_features;
}

classifier_struct* read_text_classifiers(char* filename,int& height,int& width)
{
classifier_struct* clas =NULL;
feature_struct buffer;
float err;
FILE* f = fopen(filename,"r");
fscanf(f,"Size: %d X %d\n",&height,&width);
long pos = ftell(f);
int count = 0;
int len = 0;
int num;

int type,top,bot,left,right,thresh,polar;

do
{
	len = fscanf(f,"Feature[%d]: Type: %d T: %d L: %d B: %d R: %d Thresh: %d Polar: %d Error: %f\n",&num,&type, &top, &left, &bot, &right, &thresh, &polar,&err);
	if(len>1)count++;
}
while(len>1);

clas = (struct classifier_struct *) malloc(sizeof(struct classifier_struct));
//alloc clas
fseek(f,pos,0L);

clas->features = (struct feature_struct *) malloc(count*sizeof(struct feature_struct));

for(int i=0;i<count;i++)
	{	 
	len = fscanf(f,"Feature[%d]: Type: %d T: %d L: %d B: %d R: %d Thresh: %d Polar: %d Error: %f\n",&num,&type, &top, &left, &bot, &right, &thresh, &polar,&(err));
    clas->features[i].type = type;
    clas->features[i].top = top;
    clas->features[i].bottom = bot;
    clas->features[i].left = left;
    clas->features[i].right = right;
    clas->features[i].pos_neg = polar;
    clas->features[i].thresh = thresh;
	clas->features[i].error = (double)err;
	}

clas->num_features = count;

return clas;
}