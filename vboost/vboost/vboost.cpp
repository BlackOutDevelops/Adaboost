// vboost.cpp : Defines the entry point for the console application.
//


// Format of program: face_train classifier_name positive_examples cascade_lvls
//									detection_rate false_pos_rate top_feature

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
//#include <windows.h>
#include <io.h>


namespace std {};
using namespace std;
#include <vector>

#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))

char debug = 0;

#define TOTAL_FEATURES 400000
//#define NUMBER_OF_FEATURES  1000
#define SCENE_PICTURES 20000
#define NEG_COUNT 8000
//#define NEG_COUNT 999
//#define SHIFTER 1





struct feature_struct {
    char type;			// Type of feature
    char left;			// Left boundary of feature
    char top;			// Top boundary of feature
    char right;			// Right boundary of feature
    char bottom;		// Bottom boundary of feature
    int thresh;
    char polarity;
    double error;		// Error of the feature relative to the image in question
    double linear_error;
    int pos_center,neg_center;
    int num;
};

struct feature_struct_var:feature_struct
{
double variance_pos;
double variance_neg;
double avg_pos;
double avg_neg;

double variance;
double avg;

};

struct image_struct {
    double **image;		// Image in question
    double **int_image;			// Integral image of pixel values
    int height;					// Height of image
    int width;					// Width of image
    double weight;				// Weight of either the positive or negative example image
    double sum_weight;
    double scale;				// used in the "pyramid" method
    int image_num;
};



void plot_feature(feature_struct_var* feature,image_struct* positives,int max_pos, image_struct* negatives,int max_neg);
void plot_margin(feature_struct_var* all_features,int all_feature_num, feature_struct_var* team_features,int team_feature_num);

void compute_threshold(feature_struct_var* feature,image_struct* positives,int max_pos, image_struct* negatives,int max_neg,int iteration);
void compute_threshold_tabular(feature_struct_var* feature,image_struct* positives,int max_pos, image_struct* negatives,int max_neg,int iteration,int* feature_table,int feature_table_size,int&ndx);

void compute_threshold1(feature_struct_var* feature,image_struct* pos_train,int pos_count, image_struct* neg_train,int neg_count);

void analyze_weights(feature_struct_var* feature,int feature_count);
void correct_weights(feature_struct_var* feature,int feature_count);
void compute_errors(feature_struct_var* features,int feature_count,image_struct* pos_train,int pos_count, image_struct* neg_train,int neg_count);
void retrain_experts(feature_struct_var* features,int feature_count,image_struct* positives,int pos_count, image_struct* negatives,int neg_count,int iteration);
void retrain_experts_tabular(feature_struct_var* features,int feature_count,image_struct* positives,int pos_count, image_struct* negatives,int neg_count,int iteration,int* feature_table,int feature_table_size,char in_memory);

void weight_histogram(image_struct* images, int count, char* filename);
void store_weights(image_struct* images, int count, char* filename);
void load_weights(image_struct* images, int count, char* filename);
void save_weights(image_struct* positives, int pos_count,image_struct* negatives, int neg_count,int iteration);
void recall_weights(image_struct* positives, int pos_count,image_struct* negatives, int neg_count,int iteration);

void apply_data_filter(image_struct* images,int count,char* filename);
void apply_data_filter1(image_struct* images, int count, char* filename, int nmax);
void filter_data(image_struct* positives,int pos_count,image_struct* negatives,int neg_count, int iteration,int nmax);
void compute_error(feature_struct_var* feature,image_struct* positives,int max_pos, image_struct* negatives,int max_neg);

void compute_variances(feature_struct_var* features,int feature_count,image_struct* positives,int pos_count, image_struct* negatives,int neg_count);

void create_ref(feature_struct_var* h,double* ref,double& ref_mean,double& ref_var,image_struct* positives,int pos_count, image_struct* negatives,int neg_count);
double correlation(double* ref, double ref_mean, double ref_var, feature_struct_var* h,image_struct* positives,int pos_count, image_struct* negatives,int neg_count,double*& ref1,double& ref1_mean,double& ref1_var);

int generate_features_18k(int max_h,int max_w);
int generate_features_46k(image_struct* positives,int pos_count, image_struct* negatives,int neg_count);

int generate_features_type(char type,int count,image_struct* positives,int pos_count, image_struct* negatives,int neg_count);
int features_generated = 0;

void img_mean_var(image_struct* img,double& mean, double& sigma);
void img_mean_var_correction(image_struct* img,double mean, double sigma);

void tabulate_features(char* filename,feature_struct_var* features,int num_features,image_struct* positives,int max_pos, image_struct* negatives,int max_neg);
void verify_read_speed(char* filename);

int read_img_dir(image_struct* images,char* path,char* format_str,int num_imgs,int& max_h,int& max_w,int&min_h,int&min_w);

//UTIL
void alloc_int_image(int w, int h, image_struct* temp_image);
void alloc_image(int w, int h, image_struct* temp_image);
void add_spacing(image_struct& img,int dh,int dw);

void histogram(double* fplus,int first_nonzero,int last_nonzero);


// "Team" of classifiers
struct classifier_struct {
    int num_features;					// Number of features on a "team"
    struct feature_struct_var *features;	// The actual features
    double thresh;						// Face detection threshold
};

class response_vector
{
    vector <unsigned long long> responses;
    unsigned long long checksum;
public:
    response_vector();
    char GetBit(int pos);
    void SetBit(int pos,char bit);
    void QuickCompare(response_vector& rhs);
    void FullCompare(response_vector& rhs);
    void calc_checksum();

};

response_vector::response_vector()
{

    int count = TOTAL_FEATURES/(sizeof(unsigned long long)*8)+1;
    printf("count = %d \n",count);
    responses.resize(count);
    for(int i=0;i<count;i++)
    {
        //printf("i=%d \n",i);
        responses[i]=0;
    }   
    checksum = 0;

}

void response_vector::FullCompare(response_vector& rhs)
{
for(size_t i=0;i<responses.size();i++)
    {
    char bit1 = GetBit((int)i);
    char bit2 = rhs.GetBit((int)i);
    if(bit1!=bit2)
        {
        printf("Bit mismatch at %d \n",i);
        }
    }
}


void response_vector::SetBit(int pos,char bit)
{
    /*
    if(pos<100)
        printf("bit %d is %d \n",pos,bit);
    */
    char size = sizeof(unsigned long long)*8;
    char test = sizeof(unsigned long long);
    int pos_byte = pos/size;
    int pos_bit = pos % size;
    unsigned long long mask = 1;
    mask = mask<<pos_bit;
    //printf("Setting bit %d longlong %d \n", pos_bit, pos_byte);
    if(bit)
        responses[pos_byte]|=mask;
    else
        responses[pos_byte]&=~mask;
}

char response_vector::GetBit(int pos)
{
    char size = sizeof(unsigned long long)*8;
    int pos_byte = pos/size;
    int pos_bit = pos % size;
    unsigned long long mask = 1;
    mask = mask<<pos_bit;
    unsigned long long bit = responses[pos_byte]&mask;
    if(bit) return 1;
    return 0;
}

void response_vector::calc_checksum()
{
unsigned long long sum = 0;
for(size_t i=0;i<responses.size();i++)
    {
    unsigned long long curr = responses[i];
    sum+=responses[i];
    }

checksum = sum;

printf("checksum = %x\n",checksum);
}

struct feature_struct_var     feature_set[TOTAL_FEATURES];		// Entire set of features
struct classifier_struct *classifiers;				// Number of classifiers in the cascade.
int abs_deltathresh_max=0;
int* feature_table = NULL;
size_t feature_table_size = 40*1000*1000;
size_t memory_table_size = 4*10000*32000;

double **load_image(char *filename,int *height,int *width);
void integral_image(double **input, double **output, int width, int height);



int apply_feature(double**image, struct feature_struct_var* feature, int x, int y);
//void write_image(unsigned char **image, char *filename, int height, int width);
int apply_thresh_feature(double**image, struct feature_struct_var* feature, int x, int y);
void unload(struct image_struct old);
void unload_temp(struct image_struct old);
int apply_thresh_class(struct classifier_struct classifier, double**image,int x, int y);
void quicksort_images_weight(struct image_struct v [], int n);
//unsigned char **extract_image(unsigned char **image, int x, int y, int height, int width);
void save_classifiers(classifier_struct* classifiers,int iteration,int clas_loop);
void load_classifiers(classifier_struct* classifiers,int iteration,int clas_loop);


/* swap:  interchange v[i] and v[j]. */
void swap_images(struct image_struct v[], int i, int j)
{
    struct image_struct temp;

    temp = v[i];
    v[i] = v[j];
    v[j] = temp;
}


/* quicksort: sort v[0]..v[n-1] into increasing order. */
void quicksort_images(struct image_struct v [], int n)
{
    int i, last;
    if (n <= 1)                         /* nothing to do */
        return;
    swap_images(v,0,rand() % n);       /* move pivot element to v[0] */
    last = 0;
    for (i = 1; i < n; i++)         /* partition */
        if (v[i].sum_weight<v[0].sum_weight)
            swap_images(v,++last, i);
    swap_images(v, 0, last);                 /* restore pivot */
    quicksort_images(v,last);               /* recursively sort each part. */
    quicksort_images(v+last+1, n-last-1);
}

void quicksort_images_weight(struct image_struct v [], int n)
{
    int i, last;
    if (n <= 1)                         /* nothing to do */
        return;
    swap_images(v,0,rand() % n);       /* move pivot element to v[0] */
    last = 0;
    for (i = 1; i < n; i++)         /* partition */
        if (v[i].weight>v[0].weight)
            swap_images(v,++last, i);
    swap_images(v, 0, last);                 /* restore pivot */
    quicksort_images_weight(v,last);               /* recursively sort each part. */
    quicksort_images_weight(v+last+1, n-last-1);
}

//vector <response_vector> cvect;
//response_vector vect;
//response_vector old;

#define HIST_LEN 30000

double    hist_pos2[HIST_LEN];
double    hist_neg2[HIST_LEN];
double    fplus[HIST_LEN];
double    fminus[HIST_LEN];

char* positive_path = "../positive";
char* positive_mask = "../positive/positive%04d.pgm";

char* negative_path = "../scenery";
char* negative_mask = "../scenery/scenery%05d.pgm";



int main(int argc, char* argv[]) {
    
    //array's holding training images
    struct image_struct *pos_train;	// Positive set of images
    struct image_struct *neg_train;	// Negative set of images

    //support variables for training images
    int   pos_count;		// Number of positive examples
    int   neg_count;		// Number of negative examples

    //temp variables
    int    x, y, z1, z2;
    //int z;
    //int a;
    //int	   com_neg;
    //int    com_pos;
/*
    int    begin;
    int    end;

    int    misclassneg, misclassneg2;
    int    misclasspos, misclasspos2;
*/
    int    topfeat;
    //float    pos_checker, neg_checker;
    char   filename[80];
    char  *clas_name;
    //	char  *weights;
   // double misclass;
    double weight_balance;
    double d;
    double f;
    //double scale;
    //double max_scale;
    FILE  *output;
    //	FILE  *weightlist;
    //struct image_struct temp_image;

    //main control loop variables
    int clas_loop;			// Classifier control loop
    int ada_loop;			// AdaBoost feature-finding loop
    int classifier_lvls;	// Number of classifier levels
    //double detection_rate;	// Overall detection rate
    //double false_pos_rate;	// False negative rate per layer

    //Histogram variables for finding feature threshold
    
    //int    max_error_thresh;
    //int    min_error_thresh;
    //double max_error;
    //double min_error;

    
    
    //Histogram variables for finding classifier threshold

    // Name of the classifier group (i.e., new.clas)
    clas_name = argv[1];

    // Number of positive images.
    pos_count = atoi(argv[2]);

    // Initialize the negative image counter.
    neg_count = atoi(argv[3]);

    // Number of classifier levels in the cascade.
    classifier_lvls = atoi(argv[4]);

    // Max # of features.
    topfeat = atoi(argv[5]);

	char tabulate_only = 0;
	char verify_only = 0;
	char tabular = 0;
	char in_memory = 0;

	int max_height = 0;
	int max_width = 0;
	int min_height=24,min_width=24;
	int dummy;

	char eat_whole_directory = 0;

	char data_filter = 0;
	int filter_iteration = 0;

	if(argc>=7 )
		{
		if(!stricmp(argv[6],"-t"))
			{
			tabulate_only = 1;
			}
		else if(!stricmp(argv[6],"-v"))
			{
			verify_only = 1;
			}
		else if(!stricmp(argv[6],"-tab"))
			{
			tabular = 1;
			feature_table = (int*)malloc(feature_table_size);
			}
		else if(!stricmp(argv[6],"-mem"))
			{
			tabular = 1;
			in_memory = 1;
			feature_table = (int*)malloc(memory_table_size);
			}
		else if(!stricmp(argv[6],"-eyefull"))
			{
			eat_whole_directory = 1;
			//positive_path = "../eye-pos/full/";
			positive_path = "../eyes/full/positives/";
			positive_mask = "../eyes/full/positives/*.pgm";

			negative_path = "../eyes/full/negatives/";
			negative_mask = "../eyes/full/negatives/*.pgm";
			}

		if(argc>=8 && (!stricmp(argv[7],"-filter")))
			{
			filter_iteration = atoi(argv[8]);
			data_filter = 1;
			}
		
		}

    // Allocates space for the classifiers.
    classifiers = (struct classifier_struct *) malloc(classifier_lvls*sizeof(struct classifier_struct));

    // Generate the entire feature list.
    srand((unsigned int)time(0));

	//toDo:compute max_h, max_w
    
    
    // Load Positive Training Examples
    // Malloc space for the training images.
    pos_train = (struct image_struct *) malloc(pos_count*sizeof(struct image_struct));
    //neg_count = NEG_COUNT;
    // Load the images.

    /*
    HANDLE hprocess = GetCurrentProcess();
    SetPriorityClass(hprocess,REALTIME_PRIORITY_CLASS);
    HANDLE hthread = GetCurrentThread();
    SetThreadPriority(hthread,THREAD_PRIORITY_TIME_CRITICAL);
    int pr = GetThreadPriority(hthread);
    */

	if(eat_whole_directory)
		{
		pos_count = read_img_dir(pos_train,positive_path,positive_mask,pos_count,max_height,max_width,min_height,min_width);
		neg_train = (struct image_struct *) malloc(neg_count * sizeof(struct image_struct));
		neg_count = read_img_dir(neg_train,negative_path,negative_mask,neg_count,max_height,max_width,dummy,dummy);

		int i;


		for(i =0 ;i<pos_count;i++)
			{
			pos_train[i].height = min_height;
			pos_train[i].width = min_width;
			}

		
        
		int dummy;

		
		//crop all negative images to (max_height,max_width)
		for(i=0;i<neg_count;i++)
			{
			//memory leak here !
		    //should use crop_image to prevent it
			neg_train[i].height = min_height;
			neg_train[i].width = min_width;
			}

		//allocate and compute
		//integral images
		for(i=0;i<pos_count;i++)
			{
			alloc_int_image(pos_train[i].width,pos_train[i].height,&(pos_train[i]));
			integral_image(pos_train[i].image,pos_train[i].int_image,pos_train[i].width,pos_train[i].height);
			pos_train[i].weight = 1.0/(pos_count+pos_count);
			pos_train[i].sum_weight = pos_train[i].weight;
			pos_train[i].image_num = i;
			}
		for(i=0;i<neg_count;i++)
			{
			alloc_int_image(neg_train[i].width,neg_train[i].height,&(neg_train[i]));
			integral_image(neg_train[i].image,neg_train[i].int_image,neg_train[i].width,neg_train[i].height);
			neg_train[i].weight = 1.0/(neg_count+neg_count);
			neg_train[i].sum_weight = neg_train[i].weight;
			neg_train[i].image_num = i;
			}

		
		char debug  =1;
		}
//else
	else
		{
		for(y = 0;y < pos_count;y++) 
			{
			z1=y+1;
			sprintf(filename,"../positive/positive%04d.pgm",y+1);
			pos_train[y].image_num = y+1;
			pos_train[y].image = load_image(filename,&(pos_train[y].height),&(pos_train[y].width));
			pos_train[y].int_image = NULL;
			double mean,sigma;
			//img_mean_var(&pos_train[y],mean,sigma);
			//img_mean_var_correction(&pos_train[y],mean,sigma);
			pos_train[y].scale = 1;			// Initial scale of the image.

			// Make space for the integral image.
			pos_train[y].int_image = (double **) malloc((pos_train[y].height+1)*sizeof(double *));
			for(x = 0;x <= pos_train[y].height;x++) {
				pos_train[y].int_image[x]=(double *) malloc((pos_train[y].width+1)*sizeof(double));
				}

			// Calculate integral image.
			integral_image(pos_train[y].image,pos_train[y].int_image,pos_train[y].width,pos_train[y].height);

			// Initialize weights
			pos_train[y].weight=1.0/((double) (pos_count+pos_count));
			//pos_train[y].weight=1.0/((double) (pos_count+10*neg_count));
			pos_train[y].sum_weight=pos_train[y].weight;
			if(pos_train[y].height>max_height) max_height = pos_train[y].height;
			if(pos_train[y].width>max_width) max_width = pos_train[y].width;
			//		printf("image %d processed\n", y);
			
			}
		

		//Load Negative Training Examples


		neg_train = (struct image_struct *) malloc(neg_count * sizeof(struct image_struct));
		for(y = 0;y < neg_count; y++) {
			if(y%10 == 0) printf("loading neg image: %d\n",y);

			//z1 = (rand() % SCENE_PICTURES) + 1; ///AlexK let's be deterministic for now
			z1 = y+2;
			sprintf(filename,"../scenery/scenery%05d.pgm",z1);
			neg_train[y].image_num = z1;

			neg_train[y].image = load_image(filename, &(neg_train[y].height), &(neg_train[y].width));
			neg_train[y].int_image = NULL;
			double mean, sigma;
			//img_mean_var(&neg_train[y],mean,sigma);
			//img_mean_var_correction(&neg_train[y],mean,sigma);
			neg_train[y].int_image = (double **) malloc((neg_train[y].height+1)*sizeof(double *));
			for(x = 0;x <= neg_train[y].height;x++) {
				neg_train[y].int_image[x] = (double *) malloc((neg_train[y].width+1)*sizeof(double));
			}
			integral_image(neg_train[y].image,neg_train[y].int_image,neg_train[y].width,neg_train[y].height);
			neg_train[y].weight = 1.0/((double) (neg_count+neg_count));
			//neg_train[y].weight = 10.0/((double) (pos_count+10*neg_count));
			neg_train[y].sum_weight = neg_train[y].weight;
			if(neg_train[y].height>max_height) max_height = neg_train[y].height;
			if(neg_train[y].width>max_width) max_width = neg_train[y].width;
			
			}
		}

    
    //load weights
    //sort weights
    //filter out extreme weights 
    //reinit the rest of the weights

    //apply_data_filter(&pos_train[0],8,"pos-weights.txt");
    //apply_data_filter(&neg_train[0],8,"neg-weights.txt");
	if(data_filter)
		filter_data(&pos_train[0],pos_count,&neg_train[0],neg_count,filter_iteration,8);

    //filter out outliers
    //features_generated = generate_features_46k(pos_train,pos_count,neg_train,neg_count);
    
	features_generated = generate_features_18k(min_height,min_width);

    output = fopen(clas_name,"rb");
	
    if(output != NULL) {
		FILE* text_output = fopen("new.txt","w");
		fprintf(text_output,"Size: %d X %d\n",pos_train[0].height,pos_train[0].width);

        fread(&clas_loop,sizeof(int),1,output);
        printf("clas_loop: %d\n",clas_loop);
        fread(classifiers,sizeof(struct classifier_struct),clas_loop+1,output);
        for(x = 0;x <= clas_loop;x++) {
            printf("Classifier %d: Thresh: %f Features: %d\n",x,classifiers[x].thresh,classifiers[x].num_features);
            classifiers[x].features=(struct feature_struct_var *) malloc((classifiers[x].num_features)*sizeof(struct feature_struct_var));
            feature_struct temp;
            for(int i=0;i<classifiers[x].num_features;i++)
                {
                fread(&temp,sizeof(struct feature_struct),1,output);
                classifiers[x].features[i]=*((feature_struct_var*)&temp);
                }
            
            for(y=0;y<classifiers[x].num_features;y++) {
                printf("Feature[%d]: Type: %d T: %d B: %d L: %d R: %d Thresh: %d Error: %f Linerr %f\n",y,classifiers[x].features[y].type, classifiers[x].features[y].top, classifiers[x].features[y].bottom, classifiers[x].features[y].left, classifiers[x].features[y].right, classifiers[x].features[y].thresh, classifiers[x].features[y].error,classifiers[x].features[y].linear_error);
				fprintf(text_output,"Feature[%d]: Type: %d T: %d L: %d B: %d R: %d Thresh: %d Polar: %d Error: %f \n",y,classifiers[x].features[y].type, classifiers[x].features[y].top, classifiers[x].features[y].left, classifiers[x].features[y].bottom, classifiers[x].features[y].right, classifiers[x].features[y].thresh, classifiers[x].features[y].polarity,classifiers[x].features[y].error);
            }
        }
        fclose(output);
		fclose(text_output);
        double* ref = (double*)malloc((pos_count+neg_count)*sizeof(double));
        double ref_mean,ref_var,correl;
        double *ref1,ref1_mean,ref1_var;
        for(int i=0;i<classifiers[0].num_features;i++)
            for(int j=0;j<classifiers[0].num_features;j++)
                if(i<j)
                    {
                    create_ref(&classifiers[0].features[i],ref,ref_mean,ref_var,pos_train,pos_count,neg_train,neg_count);
                    correl = correlation(ref,ref_mean,ref_var,&classifiers[0].features[j],pos_train,pos_count,neg_train,neg_count,ref1,ref1_mean,ref1_var);
                    if(fabsf(correl)>0.5)
                        printf("Features: %d %d Correlation %f\n",i,j,correl);
                    free(ref1);
                    }
        free(ref);
        while(1);
        return 0;
    }
    // Pay attention here.
    else {
        clas_loop=-1;
        printf("no previous file\n");
        for(int i=0;i<classifier_lvls;i++)
            {
            classifiers[i].features = NULL;
            }
        /*
		if(!eat_whole_directory)
			neg_count = NEG_COUNT;
        */
        }
    //	weightlist = fopen(weights,"wb");
    //-------------------------------------------------------------
    // Main classifier loop.

	//load classifiers here
	int first_iteration = -1;

	char load_data = 0;

	if(argc>=7)
		{
		if(!stricmp(argv[6],"-load"))
			{
			first_iteration = atoi(argv[7]);
			load_data = 1;
			recall_weights(pos_train,pos_count,neg_train,neg_count,first_iteration);			
			}
		}

    for(clas_loop=0;clas_loop < classifier_lvls;clas_loop++) {
        printf("Class_loop: %d\n",clas_loop);

        d=0.0;
        f=100.0;
        //compute_variances(feature_set,features_generated,&pos_train[0],pos_count,&neg_train[0],neg_count);
        int old;
        classifiers[clas_loop].features = (struct feature_struct_var *) malloc((topfeat+1)*sizeof(struct feature_struct_var));
		if(load_data==1)
			load_classifiers(classifiers,first_iteration,clas_loop);

		if(tabulate_only)
			{
			tabulate_features("dbase.dat",feature_set,features_generated,&pos_train[0],pos_count,&neg_train[0],neg_count);
			while(1);
			}
		if(verify_only)
			{
			printf("verification started.....\n");
			verify_read_speed("dbase.dat");
			while(1);
			}

		if(in_memory)
			{
			FILE* f = fopen("dbase.dat","rb");
			fread(feature_table,1,memory_table_size,f);
			fclose(f);
			}
		

        for (ada_loop = first_iteration+1; ada_loop <= topfeat; ada_loop++) {

            //compute weights first
			if(!tabular)
				retrain_experts(feature_set,features_generated,&pos_train[0],pos_count,&neg_train[0],neg_count,ada_loop+1);
			else
				retrain_experts_tabular(feature_set,features_generated,&pos_train[0],pos_count,&neg_train[0],neg_count,ada_loop+1,feature_table,feature_table_size,in_memory);
            //compute_errors(feature_set,features_generated,&pos_train[0],pos_count,&neg_train[0],neg_count);
            //analyze_weights(feature_set,features_generated);

            z2 = 0;
            for(y = 1;y < features_generated;y++) {
                if(feature_set[z2].error > feature_set[y].error) {
                    old = z2;
                    z2 = y;                    
                }
            }
        //z2 = old;
            
            //compute_error(&feature_set[z2],&pos_train[0],pos_count,&neg_train[0],neg_count);
            
            printf("feature selected: %d   err:  %f \n pos_avg %f pos_var %f neg_avg %f neg_var %f \n", z2, feature_set[z2].error,feature_set[z2].avg_pos,feature_set[z2].variance_pos,feature_set[z2].avg_neg,feature_set[z2].variance_neg);
            // Creates an array of all of the "best" selected classifiers.

            classifiers[clas_loop].num_features = ada_loop + 1;
            classifiers[clas_loop].features[ada_loop] = feature_set[z2];

            printf("Feature #: %d\n", ada_loop + 1);

            // Weight balance is now used as the sum of the weights.
            weight_balance = 0.0;
            //update weights

            for(x = 0;x < pos_count;x++) {
                if(x==0 && ada_loop==0) debug = 1;
                if(apply_thresh_feature(pos_train[x].int_image,&feature_set[z2],0,0) == 1) 
                {
                    double old_weight = pos_train[x].weight;
                    pos_train[x].weight = pos_train[x].weight * (feature_set[z2].error/(1-feature_set[z2].error));
                    if(x==0)printf("positive image %d correctly classified, image weight was %10.7f now = %10.7f\n",x,old_weight,pos_train[x].weight);
                }
                else 
                {
                    if(x==0)printf("positive image %d misclassified \n",x);
                }
                weight_balance = weight_balance + pos_train[x].weight;
                //				fprintf(weightlist,"Weight of positive %d: %f\n",(x + SHIFTER),pos_train[x].weight);
                debug = 0;
            }
            for(x = 0;x < neg_count;x++) {
                if(apply_thresh_feature(neg_train[x].int_image,&feature_set[z2],0,0) == 0) {
                    neg_train[x].weight = neg_train[x].weight * (feature_set[z2].error/(1-feature_set[z2].error));
                }
                weight_balance = weight_balance + neg_train[x].weight;
                //				fprintf(weightlist,"Weight of negative %d: %f\n",(x + 1),neg_train[x].weight);
            }
            printf("%.15f\n",weight_balance);

            //rebalance weights
            for(x = 0;x < pos_count;x++) {
                pos_train[x].weight = pos_train[x].weight / weight_balance;
                pos_train[x].sum_weight+=pos_train[x].weight;
                //printf("Weight of positive %d: %f\n",x, pos_train[x].weight);
            }
            for(x = 0;x < neg_count;x++) {
                neg_train[x].weight = neg_train[x].weight / weight_balance;
                neg_train[x].sum_weight+= neg_train[x].weight;
                //printf("Weight of negative %d: %f\n",x, neg_train[x].weight);
            }

           

            weight_balance = 0.0;
            for(x = 0;x < classifiers[clas_loop].num_features;x++) {
                weight_balance += log((1.0-classifiers[clas_loop].features[x].error) / classifiers[clas_loop].features[x].error);
                //printf("log %d: %f\n",x,(classifiers[clas_loop].features[x].error));
            }
            // check current classifier error


            printf("wb for clas: %f\n",weight_balance);
            x = 0;
            y = 0;


            d = 0.0;
            f = 0.0;
            // Determine the false positive rate and positive detection rates for the system.
            // Sum the number of pictures classified correctly going up from 0 for the detection rate
            // and down for the false positive rate.

            //classifiers[clas_loop].thresh = ((double) z1)/100000;
			classifiers[clas_loop].thresh = 0.5;
			if(ada_loop%10==0)
				{
				save_classifiers(classifiers,ada_loop,clas_loop);
				save_weights(pos_train,pos_count,neg_train,neg_count,ada_loop);
				}
            //printf("%d d: %f f: %f\n", z1, d, f);

        }

       

        output = fopen(clas_name,"wb");
		if(output==NULL)
			{
			printf("Unable to open file %s\n",clas_name);
			}
        fwrite(&clas_loop,sizeof(int),1,output);
        printf("clas_loop: %d\n",clas_loop);
        fwrite(classifiers,sizeof(struct classifier_struct),clas_loop+1,output);
        


        for(x = 0;x <= clas_loop;x++) {
            printf("Classifier %d: Thresh: %f Features: %d\n",x,classifiers[x].thresh,classifiers[x].num_features);
            for(int i=0;i<classifiers[x].num_features;i++)
                {
                fwrite((feature_struct*)&(classifiers[x].features[i]),sizeof(feature_struct),1,output);
                }
            
            for(y = 0;y < classifiers[x].num_features;y++) {
                printf("Feature[%d]: Type: %d T: %d B: %d L: %d R: %d Thresh: %d Error: %f Linear Error: %f\n",y,classifiers[x].features[y].type, classifiers[x].features[y].top, classifiers[x].features[y].bottom, classifiers[x].features[y].left, classifiers[x].features[y].right, classifiers[x].features[y].thresh, classifiers[x].features[y].error,classifiers[x].features[y].linear_error);
            }
        }

        fclose(output);
        quicksort_images(pos_train,pos_count);
        quicksort_images(neg_train,neg_count);

        printf("Top 20 positive images:\n");
        for(x=0;x<20;x++)
        {
            printf("Image: %d sum weight: %10.7f\n",pos_train[x].image_num,pos_train[x].sum_weight);
        }

        printf("Top 20 negative images:\n");
        for(x=0;x<20;x++)
        {
            printf("Image: %d sum weight: %10.7f\n",neg_train[x].image_num,neg_train[x].sum_weight);
        }

        printf("Bottom 20 positive images:\n");
        for(x=0;x<20;x++)
        {
            printf("Image: %d sum weight: %10.7f\n",pos_train[pos_count-x-1].image_num,pos_train[pos_count-x-1].sum_weight);
        }

        printf("Bottom 20 negative images:\n");
        for(x=0;x<20;x++)
        {
            printf("Image: %d sum weight: %10.7f\n",neg_train[neg_count-x-1].image_num,neg_train[neg_count-x-1].sum_weight);
        }

        //now, compute cumulative weight histograms
        //for positives
        //weight_histogram(pos_train,pos_count,"positive-sum-hist.txt");
        //for negatives
        //weight_histogram(neg_train,neg_count,"negative-sum-hist.txt");

        //store sorted image weights
        //for positives
        store_weights(pos_train,pos_count,"pos-weights.txt");
        //for negatives
        store_weights(neg_train,neg_count,"neg-weights.txt");
		
    }


//histograms here
    neg_count = NEG_COUNT;
    //plot_feature(&classifiers[0].features[10],&pos_train[0],pos_count,&neg_train[0],neg_count);
    //plot_margin(feature_set,features_generated,classifiers[0].features,classifiers[0].num_features);
	if(tabular)
		free(feature_table);
    while(1);
    
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------
// Generates the entire list of features.
int generate_features_18k(int max_h,int max_w) {
    char type;
    int x1;
    int y1;
    int x2;
    int y2;
    int count = 0;

    int min_width = 1;
    int min_height = 1;



	int hpad = max_h/6;
	int wpad = max_w/6;

    // Feature 4: boxes of the (x) column, (3y) row variety.
    type = 4;
    for(x1 = 0;x1 < (max_h-hpad);x1+=2) {
        for(y1 = 0;y1 < (max_w-wpad);y1+=2) {
            for(x2 = (x1 + min_height);x2 <= (max_h-1);x2++) {
                for(y2 = (y1 + min_width);y2 <= (max_w-1);y2 += 3) {
                    feature_set[count].type = type;
                    feature_set[count].top = x1;
                    feature_set[count].left = y1;
                    feature_set[count].bottom = x2;
                    feature_set[count].right = y2;
                    count++;
                }
            }
        }
    }

    printf("generated %d features of type 4\n",count);

    // Feature 3: boxes of the (3x) column, (y) row variety.
    type = 3;
    for(x1 = 0;x1 < (max_h-hpad);x1+=2) {
        for(y1 = 0;y1 < (max_w-wpad);y1+=2) {
            for(x2 = (x1 + min_height);x2 <= (max_h-1);x2 += 3) {
                for(y2 = (y1 + min_width);y2 <= (max_w-1);y2++) {
                    feature_set[count].type = type;
                    feature_set[count].top = x1;
                    feature_set[count].left = y1;
                    feature_set[count].bottom = x2;
                    feature_set[count].right = y2;
                    count++;
                }
            }
        }
    }

    printf("generated %d features of type 3 and 4\n",count);

    // Feature 2: boxes of the (2x) column, (2y) row variety.
    type = 2;
    for(x1 = 0;x1 < (max_h-hpad);x1+=2) {
        for(y1 = 0;y1 < (max_w-wpad);y1+=2) {
            for(x2 = (x1 + min_height);x2 <= (max_h-1);x2 += 2) {
                for(y2 = (y1 + min_width);y2 <= (max_w-1);y2 += 2) {
                    feature_set[count].type=  type;
                    feature_set[count].top = x1;
                    feature_set[count].left = y1;
                    feature_set[count].bottom = x2;
                    feature_set[count].right = y2;
                    count++;
                }
            }
        }
    }

    printf("generated %d features of type 2 3 and 4\n",count);

    // Feature 1: boxes of the (x) column, (2y) row variety.
    type = 1;
    for(x1 = 0;x1 < (max_h-hpad);x1+=2) {
        for(y1 = 0;y1 < (max_w-wpad);y1+=2) {
            for(x2 = (x1 + min_height);x2 <= (max_h-1);x2++) {
                for(y2 = (y1 + min_width);y2 <= (max_w-1);y2 += 2) {
                    feature_set[count].type = type;
                    feature_set[count].top = x1;
                    feature_set[count].left = y1;
                    feature_set[count].bottom = x2;
                    feature_set[count].right = y2;
                    count++;
                }
            }
        }
    }


    printf("generated %d features of all types\n",count);
    return count;
}

//----------------------------------------------------------------------------------------------------------------------
// Generates the entire list of features.
int generate_features_46k(image_struct* positives,int pos_count, image_struct* negatives,int neg_count) {
    int count = 0;

    count = generate_features_type(4,count,positives,pos_count,negatives,neg_count);
    count = generate_features_type(3,count,positives,pos_count,negatives,neg_count);
    count = generate_features_type(2,count,positives,pos_count,negatives,neg_count);
    count = generate_features_type(1,count,positives,pos_count,negatives,neg_count);
    count = generate_features_type(0,count,positives,pos_count,negatives,neg_count);


    printf("generated %d features of all types\n",count);

    return count;
}

//-----------------------------------------------------------------------------------------------------------------------
//return 1 if classifier thinks it's a face 0 if it's not a face
int apply_thresh_class(struct classifier_struct classifier, double**image,int x, int y) {
    double score1 = 0.0;
    double score2 = 0.0;
    //	printf("%d, %d \n" , x, y);
    int z;
    for(z = 0;z < classifier.num_features;z++) {
        score1 += ((double) apply_thresh_feature(image, &classifier.features[z], x, y)) * log((1-classifier.features[z].error)/classifier.features[z].error);
        score2 += log((1-classifier.features[z].error)/classifier.features[z].error);
    }
    if(score1 >= (classifier.thresh * score2)) {
        return 1;
    }
    else {
        return 0;
    }
}
//-----------------------------------------------------------------------------------------------------------------------
//return 1 if is a face and 0 if is not a face
int apply_thresh_feature(double**image, struct feature_struct_var* feature, int x, int y) {
    if(debug)
    {
        int feature_value = apply_feature(image, feature, x, y);
        printf("appl feat thresh = %d polar  = %d value = %d\n",feature->thresh,feature->polarity,feature_value);
    }
    if (feature->polarity == 1) {
        if ((apply_feature(image, feature, x, y)) > feature->thresh) {
            return 1;
        }
        else {
            return 0;
        }
    }
    if (feature->polarity == -1) {
        if ((apply_feature(image, feature, x, y)) < feature->thresh) {
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

//----------------------------------------------------------------------------------------------------------------------
//returns the output value of the feature
int apply_feature(double**image, struct feature_struct_var* feature, int x, int y) {
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
			/*
			static long dbg;
			if(dbg++>500000)
					printf("top: %d bottom: %d left: %d right: %d i1%d i2%d i3%d i4%d i5%d i6%d\n", feature->top, feature->bottom, feature->left, feature->right,i1,i2,i3,i4,i5,i6);
			*/
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

#define DEBUG_II -1
//----------------------------------------------------------------------------------------------------------------------
// Calculates the integral image of a given positive or negative example.
void integral_image(double **input, double **output, int width, int height) {

    static int image_num = 0;
    int x;
    int y;
    double current_row_sum;
    //printf("I'm in\n");
    // Initialize the integral image to zero.

    if(image_num == DEBUG_II)
    {
        printf("Input array:\n");
        for(x=0;x<height;x++)
        {
            for(y=0;y<width;y++)
            {
                printf("%03d ",input[x][y]);
            }
            printf("\n");
        }
    }

    for(x = 0;x <= height;x++) {
        output[x][0] = 0;
    }

    for(x = 0;x <= width;x++) {
        output[0][x] = 0;
    }

    // Computes the integral image.
    for(x = 0;x < height;x++) {
        current_row_sum = 0;
        for(y = 0;y < width;y++) {
            // Adds pixel values.
            current_row_sum += input[x][y];
            // Pixel is first on row -
            if(x == 0) {
                output[x+1][y+1] = current_row_sum;
            }
            // Pixel isn't first on row.
            else {
                output[x+1][y+1] = output[x][y+1] + current_row_sum;
            }
        }
    }

    if(image_num++ == DEBUG_II)
    {
        printf("Output array:\n");
        for(x=0;x<=height;x++)
        {
            for(y=0;y<=width;y++)
            {
                printf("%03d ",output[x][y]);
            }
            printf("\n");
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------------
double **load_image(char *filename,int *height,int *width) {
    FILE *input;
    char c;
    char comment[50];
    int x, y;
    int r,g,b;
    double **image;

    input = fopen(filename,"rb");
    // Check to see if called image exists.
    if(input == NULL) {
        printf("oh shit %s\n",filename);
        exit(1);
        //		return NULL;
    }
    c = getc(input);
    // Check to make sure training image is the right ytype.
    if(c!='P') {
        fclose(input);
        return NULL;
    }
    c = getc(input);
    // Image is grayscale.
    if(c =='5') {
        fscanf(input," ");
        c = fgetc(input);
        // Find comment lines.
        while(c == '#') {
            fgets(comment,40,input);
            while(comment[strlen(comment)-1] != '\n') {
                fgets(comment,40,input);
            }
            fscanf(input," ");
            c = fgetc(input);
        }
        ungetc(c,input);
        // Read in height and width of image
        fscanf(input," %d %d ",width, height);
        c = fgetc(input);
        while(c == '#') {
            fgets(comment,40,input);
            while(comment[strlen(comment)-1] != '\n') {
                fgets(comment,40,input);
            }
            fscanf(input," ");
            c = fgetc(input);
        }
        ungetc(c,input);
        fscanf(input," %d",&x);
        fgetc(input);
        image = (double **) malloc(*height * sizeof(double *));
        for(x = 0;x < *height; x++) {
            image[x] = (double *) malloc(*width * sizeof(double));
            for(y = 0;y < *width; y++) {
                image[x][y] = (double)fgetc(input);
            }
        }
        fclose(input);
        return image;
    }
    // Not used in this program - can ignore!
    else if(c == '6') {
        fscanf(input," ");
        c = fgetc(input);
        while(c == '#') {
            fgets(comment,40,input);
            while(comment[strlen(comment)-1] != '\n') {
                fgets(comment,40,input);
            }
            fscanf(input," ");
            c = fgetc(input);
        }
        ungetc(c,input);
        fscanf(input," %d %d ",width, height);
        c = fgetc(input);
        while(c == '#') {
            fgets(comment,40,input);
            while(comment[strlen(comment)-1] != '\n') {
                fgets(comment,40,input);
            }
            fscanf(input," ");
            c = fgetc(input);
        }
        ungetc(c,input);
        fscanf(input," %d",&x);
        fgetc(input);
        image = (double **) malloc(*height * sizeof(double *));
        for(x = 0;x < *height;x++) {
            image[x] = (double *) malloc(*width * sizeof(double));
            for(y = 0;y < *width;y++) {
                r = fgetc(input);
                g = fgetc(input);
                b = fgetc(input);
                image[x][y] = (double)((222 * r) + (707 * g) + (71 * b)) /1000;
            }
        }
        fclose(input);
        return image;
    }
    else {
        fclose(input);
        return NULL;
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
		
	}

	if(old.int_image!=NULL)
	{
		free(old.int_image[x]);
		old.int_image[x]=NULL;
	}

	free(old.image);
	old.image = NULL;
	
	if(old.int_image!=NULL)
		free(old.int_image);
	old.int_image = NULL;
}

//----------------------------------------------------------------------------------------------------------------
void unload_temp(struct image_struct old) {
    int x;
    for (x = 0; x <= old.height; x++) {
        free(old.int_image[x]);
    }
    free(old.image);
    free(old.int_image);
}


void plot_feature(feature_struct_var* feature,image_struct* positives,int max_pos, image_struct* negatives,int max_neg)
{
int i;
memset(&hist_pos2[0],0,sizeof(hist_pos2));
memset(&hist_neg2[0],0,sizeof(hist_neg2));

char filename[80];
sprintf(filename, "feature_%d.txt",feature->num);

FILE* f = fopen(filename,"w");
fprintf(f,"Feature number: %d\n",feature->num);
fprintf(f,"Polarity: %d \n",feature->polarity);
fprintf(f,"Positive centroid: %d \n",feature->pos_center);
fprintf(f,"Negative centroid: %d \n",feature->neg_center);
fprintf(f,"Threshold: %d\n",feature->thresh);
fprintf(f,"Linear error: %f \n",feature->linear_error);
fprintf(f,"Weighted error: %f \n",feature->error);
int value;


for(i=0;i<max_pos;i++)
    {
    value=apply_feature(positives[i].int_image,feature,0,0) + 100000;
    if(value<0 || value >= 300000)
            {
            printf("Error ! Histogram Array Out Of Bounds\n");
            }
    else
        {
        hist_pos2[value]++;
        }
    }


for(i=0;i<max_neg;i++)
    {
    value=apply_feature(negatives[i].int_image,feature,0,0) + 100000;
    if(value<0 || value >= 300000)
        {
        printf("Error ! Histogram Array Out Of Bounds\n");
        }
    else
        {
        hist_neg2[value]++;
        }
    }

for(i=0;i<300000;i++) hist_pos2[i]= hist_pos2[i]/max_pos;
for(i=0;i<300000;i++) hist_neg2[i]= hist_neg2[i]/max_neg;
    
for(i=feature->pos_center-2000;i<feature->pos_center+2000;i++)
    {
    fprintf(f,"%f %f\n",hist_pos2[i],hist_neg2[i]);
    }

// recompute thresh

double diff = 0;
double best_diff=0;
int best_pos =0;

for(i=0;i<300000;i++)
    {
    diff = diff + hist_pos2[i]-hist_neg2[i];
    if(fabsf(diff)>=best_diff)
        {
        best_diff = fabsf(diff);
        best_pos = i;
        }
    }

fprintf(f,"Optimum thresh is at %d \n",best_pos);
feature->thresh = best_pos-100000;

// compute new error rate...
double misclass = 0;
for(i=0;i<max_pos;i++)
    {
    int res = apply_thresh_feature(positives[i].int_image,feature,0,0);
    if(res!=1)
        {
        misclass+=positives[i].weight;
        }        
    }

for(i=0;i<max_neg;i++)
    {
    int res = apply_thresh_feature(negatives[i].int_image,feature,0,0);
    if(res!=0)
        {
        misclass+=negatives[i].weight;
        }

    }

//misclass/=(max_pos+max_neg);

fclose(f);
}

void plot_margin(feature_struct_var* all_features,int all_feature_num, feature_struct_var* team_features,int team_feature_num)
{
char filename[80];
image_struct sample;
FILE* f;
f=fopen("alpha-plot.txt","w");
for(int test=1;test<1000;test++)
    {
    //load image from test set
    sprintf(filename,"../positive/positive%04d.pgm",test+2000);
    sample.image_num = test+2000;
    sample.image = load_image(filename,&(sample.height),&(sample.width));
	sample.int_image = NULL;
    sample.scale = 1;			// Initial scale of the image.

    // Make space for the integral image.
    sample.int_image = (double**) malloc((sample.height+1)*sizeof(double *));
    for(int x = 0;x <= sample.height;x++) {
        sample.int_image[x]=(double *) malloc((sample.width+1)*sizeof(double));
    }

    // Calculate integral image.
    integral_image(sample.image,sample.int_image,sample.width,sample.height);
    //compute sum(alphaihi)
    unsigned int sumhi=0;
    for(int i=0;i<all_feature_num;i++)
        {
        //compute sum(hj)
        sumhi=sumhi+apply_thresh_feature(sample.int_image,&(all_features[i]),0,0);
        }

    double sumteam = 0;
    double sumall=0;
    for(int j=0;j<team_feature_num;j++)
        {
        double epsilon = team_features[j].error;
        double alpha = log((1-epsilon)/epsilon);
        sumteam+=alpha*apply_thresh_feature(sample.int_image,&(team_features[j]),0,0);
        sumall+=alpha;
        //compute sum(alphaihi)        
        }
    //unload image
    //save to file
    fprintf(f,"%d %f\n",sumhi,(float)sumteam/sumall);
    unload(sample);
    }
fclose(f);
}
void compute_error(feature_struct_var* feature,image_struct* positives,int max_pos, image_struct* negatives,int max_neg)
{
    int i;
    // compute new error rate...
    double misclass = 0;
    for(i=0;i<max_pos;i++)
    {
        int res = apply_thresh_feature(positives[i].int_image,feature,0,0);
        if(res!=1)
        {
            misclass+=positives[i].weight;
        }        
    }

    for(i=0;i<max_neg;i++)
    {
        int res = apply_thresh_feature(negatives[i].int_image,feature,0,0);
        if(res!=0)
        {
            misclass+=negatives[i].weight;
        }

    }

    feature->error = misclass;
    feature->linear_error = feature->error;
}

void compute_threshold2(feature_struct_var* feature,image_struct* positives,int max_pos, image_struct* negatives,int max_neg,int iteration)
{
    int i;
    memset(&hist_pos2[0],0,sizeof(hist_pos2));
    memset(&hist_neg2[0],0,sizeof(hist_neg2));
    memset(&fplus[0],0,sizeof(fplus));
    memset(&fminus[0],0,sizeof(fminus));
    
    int value;

    int first_nonzero=300000;
    int last_nonzero = 0;
    double mean_pos =0;
    double meansq_pos = 0;
    double sum_pos_weight =0;
    double mean_neg = 0;
    double meansq_neg = 0;
    double sum_neg_weight = 0;
    

    for(i=0;i<max_pos;i++)
    {
        value=apply_feature(positives[i].int_image,feature,0,0) + 100000;

        if(value<0 || value >= 300000)
        {
            printf("Error ! Histogram Array Out Of Bounds\n");
        }
        else
        {
            if(value<first_nonzero) first_nonzero=value;
            if(value>last_nonzero) last_nonzero = value;

            //hist_pos2[value]++;
            hist_pos2[value]+=positives[i].weight;
            mean_pos += value*positives[i].weight;
            meansq_pos+= (double)(value)*value*positives[i].weight;
            sum_pos_weight+=positives[i].weight;
        }
    }


    for(i=0;i<max_neg;i++)
    {
        value=apply_feature(negatives[i].int_image,feature,0,0) + 100000;
        if(value<0 || value >= 300000)
        {
            printf("Error ! Histogram Array Out Of Bounds\n");
        }
        else
        {
            if(value<first_nonzero) first_nonzero=value;
            if(value>last_nonzero) last_nonzero = value;

            //hist_neg2[value]++;
            hist_neg2[value]+=negatives[i].weight;
            mean_neg += value*negatives[i].weight;
            meansq_neg +=(double)(value)*value*negatives[i].weight;
            sum_neg_weight +=negatives[i].weight;
        }
    }
    

    // recompute thresh
    mean_neg/=sum_neg_weight;
    mean_pos/=sum_pos_weight;
    meansq_pos/=sum_pos_weight;
    meansq_neg/=sum_neg_weight;
    
    meansq_pos= sqrt(meansq_pos-mean_pos*mean_pos);
    meansq_neg= sqrt(meansq_neg-mean_neg*mean_neg);

    double pos_lbound = max(first_nonzero,mean_pos-5*meansq_pos);
    double pos_rbound = min(last_nonzero,mean_pos+5*meansq_pos);

    double neg_lbound = max(first_nonzero,mean_neg-5*meansq_neg);
    double neg_rbound = min(last_nonzero,mean_neg+5*meansq_neg);

    double diff = 0;
    double best_diff1=0;
    int best_pos1 =0;
    double best_diff2=0;
    int best_pos2 =0;

    for(i=int(neg_lbound);i<=int(neg_rbound);i++)
        {
        diff = diff + hist_neg2[i];
        fplus[i]=diff;
        }
    
    diff = 0;
    for(i=int(pos_rbound);i>=int(pos_lbound);i--)
        {
        diff=diff+hist_pos2[i];
        fminus[i]=diff;
        }

    best_diff1= 100000;
    best_diff2= -100000;
    //best_diff1 = best_diff2 = fplus[first_nonzero]+fminus[first_nonzero];
    best_pos1 = best_pos2 = first_nonzero;

    for(i=first_nonzero+1;i<=last_nonzero;i++)
        {
        diff = fplus[i]+fminus[i];
        if(diff<best_diff1)
            {
            best_diff1 = diff;
            best_pos1 = i;
            }

        if(diff>best_diff2)
            {
            best_diff2 = diff;
            best_pos2 = i;
            }
        
        }

    
    if(best_diff1<(1-best_diff2))
        {
        feature->pos_center = best_pos1;
        feature->polarity = -1;
        feature->thresh = best_pos1-99999;
        feature->error = best_diff1;
        }
    else
        {
        feature->pos_center = best_pos2;
        feature->polarity = 1;
        //feature->thresh = 99999-best_pos2;
        feature->thresh = best_pos2-99999;
        feature->error = 1-best_diff2;
        }


    if(feature->num%1000==0)
        {
        printf("num %d mean_pos %f mean_neg %f first_nonzero %d last_nonzero %d\n",feature->num,mean_pos,mean_neg,first_nonzero,last_nonzero);
        printf("sigmapos %f sigmaneg %f \n",meansq_pos,meansq_neg);
        }
        
    // compute new error rate...

    double misclass = 0;
    for(i=0;i<max_pos;i++)
    {
        int res = apply_thresh_feature(positives[i].int_image,feature,0,0);
        if(res!=1)
        {
            misclass+=positives[i].weight;
        }        
    }

    for(i=0;i<max_neg;i++)
    {
        int res = apply_thresh_feature(negatives[i].int_image,feature,0,0);
        if(res!=0)
        {
            misclass+=negatives[i].weight;
        }

    }

    feature->error = misclass;

    if(iteration==1)
        feature->linear_error = feature->error;
}


void compute_threshold1(feature_struct_var* feature,image_struct* pos_train,int pos_count, image_struct* neg_train,int neg_count)
{
    double misclass;
    
    double min_error,pos_checker,neg_checker;
    int com_pos,com_neg;
    int begin,a,end,z;

    memset(hist_pos2,0,sizeof(hist_pos2));
    memset(hist_neg2,0,sizeof(hist_neg2));

    //			cur_error = 0;
    //			printf("HI\n");
    for(int x = 0;x < pos_count;x++) {
        int array_ndx = apply_feature(pos_train[x].int_image,feature,0,0)+100000;
        hist_pos2[array_ndx]++;      
        }
    for(int x = 0;x < neg_count;x++) {
        int array_ndx = apply_feature(neg_train[x].int_image,feature,0,0)+100000;
        hist_neg2[array_ndx]++;
        }
    //			printf("hi\n");
    min_error = 0.5;
    pos_checker = 0;
    neg_checker = 0;
    for(int x = 0;x < 300000;x++) {
        pos_checker += (hist_pos2[x] * x);
        neg_checker += (hist_neg2[x] * x);
        }			
    // Find the centers of mass for each histogram.

    com_pos = int(pos_checker / pos_count);
    com_neg = int(neg_checker / neg_count);


    // Determine the polarity of the histograms.  (We're looking for
    // a setup where the negative histogram is in "front" of the positive one.)
    if (com_neg < com_pos) {
        feature->polarity = 1;
    }
    else if (com_pos < com_neg) {
        feature->polarity = -1;
    }
    else 
    {
        feature->polarity = 1;
    }

    // Define a space in the histogram that extends between the center of masses.
    if (feature->polarity == 1) {
        begin = com_pos;
        end   = com_neg;
    }
    if (feature->polarity == -1) {
        begin = com_pos;
        end   = com_neg;
    }
    if (begin > end) {
        a = begin;
        begin = end;
        end = a;
    }

    // Set the number of misclassified examples as 0.  Run through finding the number
    // of misclassified examples (based on polarity); if they're less than the current
    // number of misclassified examples, then set the threshold of the feature at that point.
    // Also redefine the minimum error at that point.


    feature->thresh = begin;
    for (z = begin; z <= end; z++) {
        misclass = 0;
        if (feature->polarity == 1) {
            for (a = z; a >= begin; a--) {
                misclass += hist_pos2[a];
            }
            for (a = z; a <= end; a++) {
                misclass += hist_neg2[a];
            }

            //printf("  misclass = %5.3f \n",misclass);
            if (((double) (misclass / (pos_count + neg_count))) < min_error) {
                min_error = ((double) (misclass / (pos_count + neg_count)));
                feature->thresh = z;
                //printf("  new min_error %5.3f found at %d\n",min_error,z);
            }
        }
        else if (feature->polarity == -1) {
            for (a = z; a >= begin; a--) {
                misclass += hist_neg2[a];
            }
            for (a = z; a <= end; a++) {
                misclass += hist_pos2[a];
            }

            //printf("  misclass = %5.3f \n",misclass);

            if (((double) (misclass / (pos_count + neg_count))) < min_error) {
                min_error = ((double) (misclass / (pos_count + neg_count)));
                feature->thresh = z;
                //printf("  new min_error %5.3f found at %d\n",min_error,z);
            }
        }
    }
    if (feature->polarity == 1) {
        feature->thresh = feature->thresh - 100000;
    }
    else {
        feature->thresh = 99999 - feature->thresh;
    }

    //AlexK thresh to 0
    /*
    if(fabsf(feature_set[y].thresh)>100)
    feature_set[y].thresh = 0;
    */

    // Calculate the error.
    feature->error = 0;
    char resp;



    //		if (feature_set[y].polarity == 1) {
    for	(int x = 0; x < neg_count; x++) {
        resp = apply_thresh_feature(neg_train[x].int_image,feature,0,0);
        if (resp == 1) {
            //							misclassneg++;
            feature->error += neg_train[x].weight;
        }
        //fills up response vector here 
        //vect.SetBit(x,resp);
    }
    for (int x = 0; x < pos_count; x++) {
        resp = apply_thresh_feature(pos_train[x].int_image,feature,0,0);
        if (resp == 0) {
            //							misclasspos++;
            feature->error += pos_train[x].weight;
        }

        //vect.SetBit(neg_count+x,resp);
    }

    //printf("Error: %f Thresh %d\n", feature->error,feature->thresh);
    
        feature->linear_error = feature->error;
}

void analyze_weights(feature_struct_var* feature,int feature_count)
{
int wrong_feat = 0;
for(int i=0;i<feature_count;i++)
    {
    if(feature[i].error>=0.5)
        {
        wrong_feat++;
        }
    }
printf("Invalid weak learners: %d \n",wrong_feat);
}

void correct_weights(feature_struct_var* feature,int feature_count)
{
    for(int i=0;i<feature_count;i++)
    {
        if(feature[i].error>=0.5)
            {
            feature[i].polarity = -feature[i].polarity;
            }
    }
}

void retrain_experts(feature_struct_var* features,int feature_count,image_struct* positives,int pos_count, image_struct* negatives,int neg_count,int iteration)
{

printf("Retraining experts \n");
feature_struct_var* feature;
for(int i=0;i<feature_count;i++)
    {
    if(i%1000 == 0)
        {
        printf("Feature %d\n",i);
        }
    //retrain each individual expert
    feature = &features[i];
    feature->num = i;
	if(i==8766)
		{
		char debug;
		debug =1;
		}
    compute_threshold(feature,positives,pos_count,negatives,neg_count,iteration);
    if(feature->error>=0.5)
        {
        feature->polarity = -feature->polarity;
        feature->error = 1.0-feature->error;
        }
    }
    printf("Max threshold change %d\n",abs_deltathresh_max);
}



void retrain_experts_tabular(feature_struct_var* features,int feature_count,image_struct* positives,int pos_count, image_struct* negatives,int neg_count,int iteration,int* feature_table,int feature_table_size,char in_memory)
{
	FILE* f = NULL;
	if(!in_memory) f= fopen("dbase.dat","rb");
	printf("Retraining experts \n");
	feature_struct_var* feature;
	int ndx = 0;

	
	for(int i=0;i<feature_count;i++)
	{
		if(i%1000 == 0 )
			{
			printf("Feature %d\n",i);
			if(!in_memory)
				{
				fread(feature_table,1,feature_table_size,f);
				ndx = 0;
				}
			
			//load next 40 meg
			}
		//retrain each individual expert
		feature = &features[i];
		feature->num = i;
		//compute_threshold(feature,positives,pos_count,negatives,neg_count,iteration);
		compute_threshold_tabular(feature,positives,pos_count,negatives,neg_count,iteration,feature_table,feature_table_size,ndx);
		if(feature->error>=0.5)
			{
			feature->polarity = -feature->polarity;
			feature->error = 1.0-feature->error;
			}
	}

	printf("Max threshold change %d\n",abs_deltathresh_max);
	if(!in_memory)
		fclose(f);
}

void compute_errors(feature_struct_var* features,int feature_count,image_struct* positives,int pos_count, image_struct* negatives,int neg_count)
{
feature_struct_var* feature;
int i;

for(int j=0;j<feature_count;j++)
    {
        // compute new error rate...
        feature = &features[j];
        double misclass = 0;
        for(i=0;i<pos_count;i++)
        {
            int res = apply_thresh_feature(positives[i].int_image,feature,0,0);
            if(res!=1)
            {
                misclass+=positives[i].weight;
            }        
        }

        for(i=0;i<neg_count;i++)
        {
            int res = apply_thresh_feature(negatives[i].int_image,feature,0,0);
            if(res!=0)
            {
                misclass+=negatives[i].weight;
            }

        }

        feature->error = misclass;
        if(feature->error>=0.5)
            {
            feature->polarity = -feature->polarity;
            feature->error = 1.0-feature->error;
            }
        feature->linear_error = misclass;
    }
}

void weight_histogram(image_struct* images, int count, char* filename)
{
double min_weight = images[0].sum_weight;
double max_weight = images[count-1].sum_weight;
int num_bins = 200;
double step = (max_weight-min_weight)/num_bins;
double* freq= (double*)malloc((num_bins+1)*sizeof(double));
int bin;
int i;
for(i=0;i<num_bins;i++) freq[i]=0;
for(i=0;i<count;i++)
    {
    bin = (int)((images[i].sum_weight-min_weight)/step);
    freq[bin]++;
    }
FILE* f = fopen(filename,"w");
for(i=0;i<num_bins;i++)
    {
    double x = min_weight+i*step;
    fprintf(f,"%f %f\n",x,freq[i]);
    }
fclose(f);
free(freq);
}
void store_weights(image_struct* images, int count, char* filename)
{
FILE* f =fopen(filename,"w");
for(int i=0;i<count;i++)
    {
    fprintf(f,"%d %e %e\n",images[i].image_num,images[i].weight,images[i].sum_weight);
    }

fclose(f);
}

int findmax_sum(image_struct* images, int count)
{
int pos = 0;
double max_sum = images[pos].sum_weight;
for(int i=1;i<count;i++)
	{
	if(images[i].sum_weight>max_sum)
		{
		max_sum = images[i].sum_weight;
		pos = i;
		}
	}

return pos;
}
void apply_data_filter1(image_struct* images, int count, char* filename, int nmax)
{
//load weights
load_weights(images,count,filename);
for(int i =0;i<nmax;i++)
	{
	int pos = findmax_sum(images,count);
	images[pos].sum_weight = images[pos].weight = 0;

	}

for(int i=0;i<count;i++)
	{
	if(images[i].sum_weight>0)
		images[i].weight = images[i].sum_weight = 1.0/(2*(count-nmax));
	}
}

void filter_data(image_struct* positives,int pos_count,image_struct* negatives,int neg_count, int iteration,int nmax)
{
char pos_filename[80],neg_filename[80];
sprintf(pos_filename,"weights_base/iteration%d_pos.txt",iteration);
sprintf(neg_filename,"weights_base/iteration%d_neg.txt",iteration);
apply_data_filter1(positives,pos_count,pos_filename,nmax);
//apply_data_filter1(negatives,neg_count,neg_filename,nmax);
}


void load_weights(image_struct* images, int count, char* filename)
{
	FILE* f =fopen(filename,"r");
	int num;
	float weight,cum_weight;

	for(int i=0;i<count;i++)
	{
		fscanf(f,"%d %e %e\n",&num,&weight,&cum_weight);
		images[i].image_num = num;
		images[i].weight = weight;
		images[i].sum_weight = cum_weight;
	}

	fclose(f);
}

void apply_data_filter(image_struct* images,int count,char* filename)
{
int offset;
int len;
int img_num;
float weight;

int pos;

if(!stricmp(filename,"pos-weights.txt"))
    {
    offset = 1;
    len = 2000;
    }
else if(!stricmp(filename,"neg-weights.txt"))
    {
    offset = 2;
    len = 8000;
    }

FILE* f = fopen(filename,"r");

pos = 0;
double wt;

while(!feof(f))
    {
    fscanf(f,"%d %f %f\n",&img_num,&wt,&weight);
    if((len-pos)<=count)
        {
        images[img_num-offset].sum_weight=0;
        images[img_num-offset].weight=0;
        }
    pos++;
    }
fclose(f);
}

void compute_variance(feature_struct_var* feature,image_struct* positives,int pos_count, image_struct* negatives,int neg_count)
{
double sum=0;
double sum_sq=0;
int i;
double feature_value;

for(i=0;i<pos_count;i++)
    {
    feature_value=apply_feature(positives[i].int_image,feature,0,0);
    sum+=feature_value;
    sum_sq+=feature_value*feature_value;
    }
sum_sq/=pos_count;
sum/=pos_count;
feature->variance_pos = sqrt(sum_sq-sum*sum);
feature->avg_pos = sum;
sum=0;
sum_sq=0;

for(i=0;i<neg_count;i++)
    {
    feature_value=apply_feature(negatives[i].int_image,feature,0,0);
    sum+=feature_value;
    sum_sq+=feature_value*feature_value;
    }
sum_sq/=(neg_count);
sum/=(neg_count);
feature->variance_neg = sqrt(sum_sq-sum*sum);
feature->avg_neg = sum;
}

void compute_variance1(feature_struct_var* feature,image_struct* positives,int pos_count, image_struct* negatives,int neg_count)
{
    double sum=0;
    double sum_sq=0;
    int i;
    double feature_value;

    for(i=0;i<pos_count;i++)
        {
        feature_value=apply_feature(positives[i].int_image,feature,0,0);
        sum+=feature_value*positives[i].weight;
        sum_sq+=feature_value*feature_value*positives[i].weight;
        }


    for(i=0;i<neg_count;i++)
        {
        feature_value=apply_feature(negatives[i].int_image,feature,0,0);
        sum+=feature_value*negatives[i].weight;
        sum_sq+=feature_value*feature_value*negatives[i].weight;
        }

  
    feature->variance = sqrt(sum_sq-sum*sum);
    feature->avg = sum;
}

void compute_variances(feature_struct_var* features,int feature_count,image_struct* positives,int pos_count, image_struct* negatives,int neg_count)
{
int i;
for(i=0;i<feature_count;i++)
    {
    compute_variance(&features[i],positives,pos_count,negatives,neg_count);
    if(i%100==0)
        {
        printf("Feature %d avg_pos %f variance_pos %f avg_neg %f variance_neg %f\n",i,features[i].avg_pos,features[i].variance_pos,features[i].avg_neg,features[i].variance_neg);
        }
    }
}

void create_ref(feature_struct_var* h,double* ref,double& ref_mean,double& ref_var,image_struct* positives,int pos_count, image_struct* negatives,int neg_count)
{
double sum=0;
double sum_sq = 0;
double value;
int i;
for(i=0;i<pos_count;i++)
    {
    value = apply_feature(positives[i].int_image,h,0,0);
    sum+=value*positives[i].sum_weight;
    sum_sq+=double(value)*value*positives[i].sum_weight;
    ref[i]=value;
    }

for(i=0;i<neg_count;i++)
    {
    value = apply_feature(negatives[i].int_image,h,0,0);
    sum+=value*negatives[i].sum_weight;
    sum_sq+=double(value)*value*negatives[i].sum_weight;
    ref[i+pos_count]=value;
    }

ref_mean = sum;
ref_var = sqrt(sum_sq-sum*sum);
}

double correlation(double* ref, double ref_mean, double ref_var, feature_struct_var* h,image_struct* positives,int pos_count, image_struct* negatives,int neg_count,double*& ref1,double& ref1_mean,double& ref1_var)
{
ref1= (double*)malloc((neg_count+pos_count)*sizeof(double));

int i;
double exy=0;
double res;

create_ref(h,ref1,ref1_mean,ref1_var,positives,pos_count,negatives,neg_count);

for(i=0;i<pos_count;i++)
    {
    //compute E(xy)
    exy+=ref[i]*ref1[i]*positives[i].sum_weight;
    }
for(i=0;i<neg_count;i++)
    {
    //compute E(xy)
    exy+=ref[i+pos_count]*ref1[i+pos_count]*negatives[i].sum_weight;
    }

res = (exy-ref1_mean*ref_mean)/(ref1_var*ref_var);
return res;
}

int generate_features_type(char type,int count,image_struct* positives,int pos_count, image_struct* negatives,int neg_count) 
{
    int x1;
    int y1;
    int x2;
    int y2;

    int min_width = 1;
    int min_height = 1;

    double* ref= NULL;
    double ref_mean,ref_var;
    double* ref1;
    double ref1_mean,ref1_var;
    double* ref2;
    double ref2_mean,ref2_var;
    double corr;

    int tot_cnt = 0;

    int first_cnt=count;


    // Feature 4: boxes of the (x) column, (3y) row variety.
    for(x1 = 0;x1 < 20;x1+=1) {
        for(y1 = 0;y1 < 20;y1+=1) {
            for(x2 = (x1 + min_height);x2 <= 23;x2+=1) {
                for(y2 = (y1 + min_width);y2 <= 23;y2+=1) {
                    tot_cnt++;
                    feature_set[count].type = type;
                    feature_set[count].top = x1;
                    feature_set[count].left = y1;
                    feature_set[count].bottom = x2;
                    feature_set[count].right = y2;

                    if(ref==NULL)
                    {
                        ref = (double*)malloc((pos_count+neg_count)*sizeof(double));
                        create_ref(&feature_set[count],ref,ref_mean,ref_var,positives,pos_count,negatives,neg_count);
                        count++;
                    }
                    else
                    {
                        corr = correlation(ref,ref_mean,ref_var,&feature_set[count],positives,pos_count,negatives,neg_count,ref1,ref1_mean,ref1_var);
                        if(fabsf(corr)<0.95)
                        {
                            for(int i=count-2;i>=first_cnt;i--)
                            {
                                int distance = fabsf(feature_set[count].top-feature_set[i].top)
                                    +fabsf(feature_set[count].left-feature_set[i].left)+
                                    fabsf(feature_set[count].bottom-feature_set[i].bottom)+
                                    fabsf(feature_set[count].right-feature_set[i].right);
                                if(distance<9)
                                {
                                    corr = correlation(ref1,ref1_mean,ref1_var,&feature_set[i],positives,pos_count,negatives,neg_count,ref2,ref2_mean,ref2_var);
                                    free(ref2); //AlexK- find a better use for ref2
                                    //compute correlation
                                    if(fabsf(corr)>=0.95) break;
                                }

                            }
                            //store new feature
                        }

                        if(fabsf(corr)<0.95)
                        {
                            count++;
                            free(ref);
                            ref = ref1;
                            ref_var = ref1_var;
                            ref_mean = ref1_mean;
                            //increment count
                            //create new ref point
                        }
                        else
                        {
                            free(ref1);
                        }
                    }
                }
            }
        }
    }
    
    if(ref) free(ref);

    printf("Features of type %d: generated %d  accepted %d\n",type,tot_cnt,count);
    return count;
}

int absolute_max_index = 0;


void compute_threshold(feature_struct_var* feature,image_struct* positives,int max_pos, image_struct* negatives,int max_neg,int iteration)
{
    int i;
    memset(&hist_pos2[0],0,sizeof(hist_pos2));
    memset(&hist_neg2[0],0,sizeof(hist_neg2));


    int value;

    int first_nonzero=HIST_LEN;
    int last_nonzero = 0;
    int hist_index;

    for(i=0;i<max_pos;i++)
    {
        value=apply_feature(positives[i].int_image,feature,0,0);
        hist_index = value/5 + HIST_LEN/2;

        if(hist_index>absolute_max_index) absolute_max_index = hist_index;

        if(hist_index<0 || hist_index >= HIST_LEN)
        {
            printf("Error ! Histogram Array Out Of Bounds\n");
        }
        else
        {
            if(hist_index<first_nonzero) first_nonzero=hist_index;
            if(hist_index>last_nonzero) last_nonzero = hist_index;

            hist_pos2[hist_index]+=positives[i].weight;
        }
    }


    for(i=0;i<max_neg;i++)
    {
        value=apply_feature(negatives[i].int_image,feature,0,0);
        hist_index = value/5 + HIST_LEN/2;
        if(hist_index>absolute_max_index) absolute_max_index = hist_index;

        if(hist_index<0 || hist_index >= HIST_LEN)
        {
            printf("Error ! Histogram Array Out Of Bounds\n");
        }
        else
        {
            if(hist_index<first_nonzero) first_nonzero=hist_index;
            if(hist_index>last_nonzero) last_nonzero = hist_index;

            hist_neg2[hist_index]+=negatives[i].weight;
        }
    }


    // recompute thresh

    double diff = 0;
    double best_diff=0;
    int best_pos =0;

    double sum_neg=0;
    double sum_pos=0;
    double m_pos=0,m_neg=0;

    for(i=first_nonzero;i<=last_nonzero;i++)
    {
        sum_neg += hist_neg2[i];
        sum_pos += hist_pos2[i];

        diff = diff + hist_pos2[i]-hist_neg2[i];
        if(fabsf(diff)>=fabsf(best_diff))
        {
            m_neg = sum_neg;
            m_pos = sum_pos;
            best_diff = diff;
            best_pos = i;
        }
    }

    

    feature->pos_center = best_pos;
    
   int old_thresh;

   if(iteration>1)
        {
        old_thresh = feature->thresh;
        }
   

    if(best_diff>0)
    {
        feature->polarity = -1;
        //feature->thresh = 10*(HIST_LEN/2-best_pos-1);
		feature->thresh = 5*(best_pos-HIST_LEN/2)+1;
    }
    else
    {
        feature->polarity = 1;
        feature->thresh = 5*(best_pos-HIST_LEN/2);
    }


    if(feature->polarity == -1)
        m_pos = 0;
    else
        m_neg= 0;


    for(i=last_nonzero;i>best_pos;i--)
    {
        if(feature->polarity == -1)
            m_pos+=hist_pos2[i];
        else
            m_neg+=hist_neg2[i];
    }

    

	double error = m_neg + m_pos;

    //feature->error = misclass;
	feature->error = error;

    if(iteration==1)
		feature->linear_error = feature->error;
    
    
    if(iteration>1)
        {
        int dthresh= fabsf(old_thresh-feature->thresh);
        if(dthresh>abs_deltathresh_max)
            abs_deltathresh_max=dthresh;
        }
}

void compute_threshold_tabular(feature_struct_var* feature,image_struct* positives,int max_pos, image_struct* negatives,int max_neg,int iteration,int* feature_table, int feature_table_size,int& ndx)
{
	int i;
	memset(&hist_pos2[0],0,sizeof(hist_pos2));
	memset(&hist_neg2[0],0,sizeof(hist_neg2));

    memset(&fplus[0],0,sizeof(fplus));


	int value;

	int first_nonzero=HIST_LEN;
	int last_nonzero = 0;
	int hist_index;

	for(i=0;i<max_pos;i++)
	{
		//value=apply_feature(positives[i].int_image,feature,0,0);
		value = feature_table[ndx++];
		hist_index = value/5 + HIST_LEN/2;

		if(hist_index>absolute_max_index) absolute_max_index = hist_index;

		if(hist_index<0 || hist_index >= HIST_LEN)
		{
			printf("Error ! Histogram Array Out Of Bounds\n");
		}
		else
		{
			if(hist_index<first_nonzero) first_nonzero=hist_index;
			if(hist_index>last_nonzero) last_nonzero = hist_index;

			hist_pos2[hist_index]+=positives[i].weight;
		}
	}


	for(i=0;i<max_neg;i++)
	{
		//value=apply_feature(negatives[i].int_image,feature,0,0);
		value=feature_table[ndx++];
		hist_index = value/5 + HIST_LEN/2;
		if(hist_index>absolute_max_index) absolute_max_index = hist_index;

		if(hist_index<0 || hist_index >= HIST_LEN)
		{
			printf("Error ! Histogram Array Out Of Bounds\n");
		}
		else
		{
			if(hist_index<first_nonzero) first_nonzero=hist_index;
			if(hist_index>last_nonzero) last_nonzero = hist_index;

			hist_neg2[hist_index]+=negatives[i].weight;
		}
	}


	// recompute thresh

	double diff = 0;
    double diff_sq=0;
	double best_diff=0;
    double best_diff_sq=0;
	int best_pos =0;

	double sum_neg=0;
	double sum_pos=0;
	double m_pos=0,m_neg=0;

	for(i=first_nonzero;i<=last_nonzero;i++)
	{
		sum_neg += hist_neg2[i];
		sum_pos += hist_pos2[i];

		diff = diff + hist_pos2[i]-hist_neg2[i];

        fplus[i] = hist_pos2[i]-hist_neg2[i]; //AlexK
        //diff_sq = diff_sq+hist_pos2[i]*hist_pos2[i]-hist_neg2[i]*hist_neg2[i];
        
		if(fabsf(diff)>=fabsf(best_diff))
		{
			m_neg = sum_neg;
			m_pos = sum_pos;
			//best_diff_sq = diff_sq;
            best_diff = diff;
			best_pos = i;
		}
	}



	feature->pos_center = best_pos;

	int old_thresh;

	if(iteration>1)
	{
		old_thresh = feature->thresh;
	}


	if(best_diff>0)
	{
		feature->polarity = -1;
		//feature->thresh = 10*(HIST_LEN/2-best_pos-1);
		feature->thresh = 5*(best_pos-HIST_LEN/2)+1;
	}
	else
	{
		feature->polarity = 1;
		feature->thresh = 5*(best_pos-HIST_LEN/2);
	}

	
	if(feature->polarity == -1)
		m_pos = 0;
	else
		m_neg= 0;
	
	

	for(i=last_nonzero;i>best_pos;i--)
	{
		if(feature->polarity == -1)
			m_pos+=hist_pos2[i];
		else
			m_neg+=hist_neg2[i];
	}



	double error = m_neg + m_pos;

	//feature->error = misclass;
	feature->error = error;

	if(iteration==1)
		feature->linear_error = feature->error;

 
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

void tabulate_features(char* filename,feature_struct_var* features,int num_features,image_struct* positives,int max_pos, image_struct* negatives,int max_neg)
{
int* buffer = (int*)malloc((max_pos+max_neg)*sizeof(int));
FILE* f = fopen(filename,"wb");
for(int i=0;i<num_features;i++)
	{
	if(i%1000==0) 
		{
		printf("Tabulating feature %d \n",i);
		}
	int j;
	for(j=0;j<max_pos;j++)
		{
		buffer[j]=apply_feature(positives[j].int_image,&features[i],0,0);
		}
	for(j=0;j<max_neg;j++)
		{
		buffer[max_pos+j]=apply_feature(negatives[j].int_image,&features[i],0,0);
		}
	size_t res = fwrite(buffer,sizeof(int)*(max_pos+max_neg),1,f);
	char debug = 1;
	}
fclose(f);
free(buffer);
}

void verify_read_speed(char* filename)
{
clock_t before_t =clock();
long num_read = 160*1024*1024; //160 MB
int* buffer = (int*)malloc(num_read);
FILE* f = fopen(filename,"rb");
size_t res = fread(buffer,1,num_read,f);
res = fread(buffer,1,num_read,f);
fclose(f);
free(buffer);
clock_t after_t=clock();
printf("%d bytes processed, time elapsed %d\n",res,(after_t-before_t));
}

int read_img_dir(image_struct* images,char* path,char* format_str,int num_imgs,int& max_h,int& max_w,int&min_h,int&min_w)
{
	int first_search = 1;
	long hfile,hfile1;
	char filename[80];

	max_h=0;max_w=0;
	int test_set;
	int actually_read = 0;

	for(test_set = 1;test_set <= num_imgs;test_set++) 
		{
		struct _finddata_t pgm_file;
		if(first_search==1)
			{
			hfile = (long)_findfirst(format_str,&pgm_file);
			if(hfile==-1L) return 0;
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

		if(test_set%100 == 0) 
			printf("loading image: %s\n",filename);


		images[actually_read].image_num = test_set;
		int w,h;
		images[actually_read].image = load_image(filename,&h,&w);
		images[actually_read].int_image = NULL;
		images[actually_read].width = w;
		images[actually_read].height = h;
		//allocate int_image
		//alloc_int_image(w,h,&(images[test_set]));
		if(w>max_w) max_w = w;
		if(h>max_h) max_h = h;

		if(w<min_w) min_w = w;
		if(h<min_h) min_h = h;

		if(images[actually_read].image) actually_read++;
		}

return actually_read;
}

void alloc_int_image(int w, int h, image_struct* temp_image)
{
	temp_image->int_image = (double **) malloc((h+1)*sizeof(double *));
	// allocate integral image rows
	for(int x = 0;x <= h;x++) temp_image->int_image[x] = (double *) malloc((w+1)*sizeof(double));
	temp_image->width = w;
	temp_image->height = h;
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

void save_classifiers(classifier_struct* classifiers,int iteration,int clas_loop)
{
char filename[80];
FILE* output;

sprintf(filename,"classifiers\iteration%d.cls",iteration);

output = fopen(filename,"wb");
if(output==NULL)
	{
	printf("Unable to open file %s\n",filename);
	return;
	}

fwrite(&clas_loop,sizeof(int),1,output);
printf("clas_loop: %d\n",clas_loop);
fwrite(classifiers,sizeof(struct classifier_struct),clas_loop+1,output);



for(int x = 0;x <= clas_loop;x++) {
	printf("Classifier %d: Thresh: %f Features: %d\n",x,classifiers[x].thresh,classifiers[x].num_features);
	for(int i=0;i<classifiers[x].num_features;i++)
	{
		fwrite((feature_struct*)&(classifiers[x].features[i]),sizeof(feature_struct),1,output);
	}
/*
	for(int y = 0;y < classifiers[x].num_features;y++) {
		printf("Feature[%d]: Type: %d T: %d B: %d L: %d R: %d Thresh: %d Error: %f Linear Error: %f\n",y,classifiers[x].features[y].type, classifiers[x].features[y].top, classifiers[x].features[y].bottom, classifiers[x].features[y].left, classifiers[x].features[y].right, classifiers[x].features[y].thresh, classifiers[x].features[y].error,classifiers[x].features[y].linear_error);
	}
*/
}

fclose(output);
}

void load_classifiers(classifier_struct* classifiers,int iteration,int clas_loop)
{
	char filename[80];
	FILE* output;

	sprintf(filename,"classifiers/iteration%d.cls",iteration);

	output = fopen(filename,"rb");
	if(output==NULL)
	{
		printf("Unable to open file %s\n",filename);
		return;
	}

	fread(&clas_loop,sizeof(int),1,output);
	printf("clas_loop: %d\n",clas_loop);
	fread(classifiers,sizeof(struct classifier_struct),clas_loop+1,output);



	for(int x = 0;x <= clas_loop;x++) {
		printf("Classifier %d: Thresh: %f Features: %d\n",x,classifiers[x].thresh,classifiers[x].num_features);
		for(int i=0;i<classifiers[x].num_features;i++)
		{
			printf("preved %d\n",i);
			fread((feature_struct*)&(classifiers[x].features[i]),sizeof(feature_struct),1,output);
		}		
	}

	fclose(output);
}

void save_weights(image_struct* positives, int pos_count,image_struct* negatives, int neg_count,int iteration)
{
char pos_filename[80],neg_filename[80];

sprintf(pos_filename,"weights/iteration%d_pos.txt",iteration);
sprintf(neg_filename,"weights/iteration%d_neg.txt",iteration);
store_weights(positives,pos_count,pos_filename);
store_weights(negatives,neg_count,neg_filename);

}

void recall_weights(image_struct* positives, int pos_count,image_struct* negatives, int neg_count,int iteration)
{
char pos_filename[80],neg_filename[80];

sprintf(pos_filename,"weights/iteration%d_pos.txt",iteration);
sprintf(neg_filename,"weights/iteration%d_neg.txt",iteration);
load_weights(positives,pos_count,pos_filename);
load_weights(negatives,neg_count,neg_filename);

}

void histogram(double* fplus,int first_nonzero,int last_nonzero)
{
FILE* f= fopen("histogram.txt","w");
for(int i=first_nonzero;i<=last_nonzero;i++)
    {
    fprintf(f,"%d %e\n",i,fplus[i]);
    }

fclose(f);

}