#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

/// Global variables
int threshold_value = 120;
int const max_value = 255;
int const max_BINARY_value = 255;
Mat src, blurred, src_gray, dst, src_copy, finalOutput, map_x, map_y;
int n_sq=100,max_n=150;

vector<vector<Point> > contours;
vector<Point> outerMostContour, topLine, bottomLine;
vector<Point> slicedTL, slicedTR, slicedBL, slicedBR;
vector<Vec4i> hierarchy;
RNG rng(12345);
Point TL,TR,BL,BR;
double topLength, bottomLength;
int topLineMiddle=0,bottomLineMiddle=0;
Point table[2][151][151];
vector<int> compression_params;

/// Function headers
void ThresholdNewValue( int, void* );
void findOuterContour();
void detectDistance();
Point findTLBorder();
double distanceP(int x1,int y1,int x2,int y2);
Point pointInBetween(int x1,int y1,int x2,int y2,double d);
void selectMiddle( int, void* );
void slice(vector<Point> line, vector<Point> &slicedLine,int startIndex, int endIndex, int n_sq, int mode);
double calcLength(vector<Point> line,int startIndex,int endIndex);
void drawLine(Mat m,vector<Point> line,int startIndex,int endIndex, cv::Scalar color, bool drawCircle);
void gridify(vector<Point> top,vector<Point> bottom,int page);
void perspectiveTransformation(int page);

int outputSizeW,outputSizeH,fullSizeH,fullSizeW;

/**
 * @function main
 */
int main( int argc, char** argv ) {

	int step=1;

	while(step<=6){
		switch(step){
			case 1:
				/// Load an image
				src = imread("E:\\Img Proc/DSC_3258.JPG", 1 );
				/// Convert the image to Gray
				cvtColor( src, src_gray, CV_RGB2GRAY );
				/// Save Option
				compression_params.push_back(CV_IMWRITE_JPEG_QUALITY );
				compression_params.push_back(70);
				namedWindow( "1", WINDOW_NORMAL );
				imshow("1", src );
				resizeWindow("1", src.cols, src.rows);
				break;
			case 2:
				//destroyAllWindows();
				/// Create Trackbar to choose type of Threshold
				namedWindow("2", WINDOW_NORMAL);
				createTrackbar( "Value","2", &threshold_value, max_value, ThresholdNewValue );
				/// Call the function to initialize
				ThresholdNewValue( 0, 0 );
				break;
			case 3:
				//destroyAllWindows();
				// Find Contour Lines
				findOuterContour();
				namedWindow( "3", WINDOW_NORMAL );
				imshow("3", src_copy );	

				break;
			case 4:
				//destroyAllWindows();
				detectDistance();
				namedWindow( "4", WINDOW_NORMAL);
				createTrackbar( "TopLine: Middle","4", &topLineMiddle, topLine.size()-1, selectMiddle );
				createTrackbar( "BottomLine: Middle","4", &bottomLineMiddle, bottomLine.size()-1, selectMiddle );
				createTrackbar( "Number of square","4", &n_sq, max_n, selectMiddle );

				selectMiddle( 0, 0 );
				break;
			case 5:
				slicedTL.clear();
				slicedTR.clear();
				slicedBL.clear();
				slicedBR.clear();
				slice(topLine, slicedTL,topLineMiddle, 0, n_sq,-1);
				slice(topLine, slicedTR,topLineMiddle, topLine.size()-1, n_sq,1);
				slice(bottomLine, slicedBL,bottomLineMiddle, 0, n_sq,-1);
				slice(bottomLine, slicedBR,bottomLineMiddle,bottomLine.size()-1, n_sq,1);
				// Page L (0)
				gridify(slicedTL,slicedBL,0);
				// Page R (1)
				gridify(slicedTR,slicedBR,1);
				namedWindow( "5", WINDOW_NORMAL);
				imshow("5", src_copy );
				break;
			case 6:
				//destroyAllWindows();
				// Set the output image size
				outputSizeW = topLength/n_sq/2;
				outputSizeH = topLength/n_sq/2*1.5;
				fullSizeH=outputSizeH*(n_sq);
				fullSizeW=outputSizeW*(n_sq);
				finalOutput = Mat::zeros(fullSizeH , fullSizeW*2, src.type() );
				map_x = Mat::zeros(fullSizeH , fullSizeW*2, CV_32FC1 );
				map_y = Mat::zeros(fullSizeH , fullSizeW*2, CV_32FC1 );
				perspectiveTransformation(0);
				perspectiveTransformation(1);
				remap(src, finalOutput, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0));
				namedWindow( "6", WINDOW_NORMAL);
				imshow("6", finalOutput );	 
				resizeWindow("6", fullSizeW, fullSizeH/2);
				printf("Cols: %d, Rows: %d",finalOutput.cols,finalOutput.rows);
			default:
				break;
		}
		while(true) {
			int c;
			c = waitKey( 20 );
			if( (char)c == '/'){
				destroyWindow(to_string(step));
				step--;
				if(step<=1) step=1;
				break;
			}
			else if( (char)c == ' '){
				destroyWindow(to_string(step));

				switch(step){
					case 1:
						imwrite("E:\\Img Proc/1Input.JPG", src,compression_params);
						break;
					case 2:
						imwrite("E:\\Img Proc/2Thresholding.JPG", dst,compression_params);
						break;
					case 3:
						imwrite("E:\\Img Proc/3Contours.JPG", src_copy,compression_params);
						break;
					case 4:
						imwrite("E:\\Img Proc/4Select Middle Line.JPG", src_copy,compression_params);
						break;
					case 5:
						imwrite("E:\\Img Proc/5Gridify.JPG", src_copy,compression_params);
						break;
					case 6:
						imwrite("E:\\Img Proc/6Output.JPG", finalOutput,compression_params);
						break;
					default:
						break;
				}

				step++;
				break;
			}
		}

	}

}


/* -----------------------C O R E   F U N C T I O N ------------------------ */

void ThresholdNewValue( int, void* ) {
	blur( src_gray, blurred, Size(src.rows/200,src.rows/200));
	threshold( blurred, dst, threshold_value, max_BINARY_value,THRESH_TOZERO);

	/* Find border */
	TL = findTLBorder();
	Point seedPoint = cvPoint(TL.x/2,TL.y/2);
	cv::Rect ccmp;
	floodFill(dst, seedPoint,  cv::Scalar(0), &ccmp,  cv::Scalar(0), cv::Scalar(1),4);
	imshow("2", dst );
}

void findOuterContour(){
	findContours( dst, contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE, Point(0, 0) );
	/// Draw contours
	Mat drawing = Mat::zeros( src.size(), CV_8UC3 );
	int maxSize=0,max=0;
	for ( int i = 0; i < (int)contours.size(); i++ ) {
		cv::Rect bbox = cv::boundingRect(contours[i]);
		cout<<"Contour " << i << " = "<< bbox.area() << endl;
		if(bbox.area()>maxSize){
			maxSize=bbox.area();
			max=i;
		}
	}
	Scalar color = Scalar( 255, 0, 0);
	src_copy=src.clone();
	if(max>-1) drawContours( src_copy, contours, max, color, 5, 8, hierarchy, 0, Point() );

	outerMostContour=contours[max];
}

void detectDistance(){
	topLine.clear();
	bottomLine.clear();
	src_copy=src.clone();
	// find iTL,iTR,iBL,iBR
	// note: outerMostContour[iTL]!=TL (almost in the same place, but differs sometimes)
	int index=0,size=outerMostContour.size();
	int iTL=0,iTR=0,iBL=0,iBR=0;
	int rows = src_copy.rows;
	int column = src_copy.cols;
	for(int i=0;i<size;i++){
		int x=outerMostContour[i].x;
		int y=outerMostContour[i].y;
		if(x + y					<=	outerMostContour[iTL].x			+ outerMostContour[iTL].y ) iTL=i;
		if(column-x + y				<=	column-outerMostContour[iTR].x	+ outerMostContour[iTR].y ) iTR=i;
		if(x+rows-y					<=	outerMostContour[iBL].x			+ rows-outerMostContour[iBL].y ) iBL=i;
		if(column-x + rows-y		<=	column-outerMostContour[iBR].x	+ rows-outerMostContour[iBR].y ) iBR=i;
	}
	
	// index: iTL -> iBL -> iBR -> iTR
	// Topline: iTL -> iTR
	// BottomLine: iBL -> iBR

	// create TopLine
	// Reverse Order
	index = iTL;
	topLength=0;
	bottomLength=0;
	while(index!=iTR){
		topLine.push_back(outerMostContour[index]);
		index=(index-1+size)%size;
	}
	topLength=calcLength(topLine,0,topLine.size()-1);
	drawLine(src_copy,topLine,0,topLine.size()-1,CV_RGB(0,0,255),false);

	// Find the middle point
	double tempL=0;
	for(int i=0;i<(int)topLine.size()-1;i++){
		tempL+=distanceP(topLine[i].x,topLine[i].y,topLine[i+1].x,topLine[i+1].y);
		if(tempL>topLength/2){
			topLineMiddle=i;
			break;
		}
	}
	cout << "TopLine: " << topLength <<endl;

	// create BottomLine
	// Normal Order
	index = iBL;
	while(index!=iBR){
		bottomLine.push_back(outerMostContour[index]);
		index=(index+1)%size;
	}
	bottomLength=calcLength(bottomLine,0,bottomLine.size()-1);
	drawLine(src_copy,bottomLine,0,bottomLine.size()-1,CV_RGB(0,0,255),false);
	// Find the middle point
	tempL=0;
	for(int i=0;i<(int)bottomLine.size()-1;i++){
		tempL+=distanceP(bottomLine[i].x,bottomLine[i].y,bottomLine[i+1].x,bottomLine[i+1].y);
		if(tempL>bottomLength/2){
			bottomLineMiddle=i;
			break;
		}
	}
	cout << "BottomLine: " << bottomLength << endl;
}

void selectMiddle( int, void* ) {
	src_copy=src.clone();
	cv::line(src_copy, topLine[topLineMiddle], bottomLine[bottomLineMiddle], CV_RGB(0,255,0), 10, 8,0);
	imshow("4", src_copy );
}

void slice(vector<Point> line, vector<Point> &slicedLine,int startIndex, int endIndex, int n_sq, int incMode){

	// incMode=-1 means reverse-Order
	double length=calcLength(line,startIndex,endIndex);
	Point lastPoint;

	lastPoint=line[startIndex];
	slicedLine.push_back(lastPoint);
	int index=startIndex+incMode;
	double remain=length/(double)(n_sq);
	while(index!=endIndex){
		double d=distanceP(lastPoint.x,lastPoint.y,line[index].x,line[index].y);
		if(remain<d){
			lastPoint=pointInBetween(lastPoint.x,lastPoint.y,line[index].x,line[index].y,remain);
			slicedLine.push_back(lastPoint);

			remain=length/(double)(n_sq);
			index+=incMode;
		}
		else{
			lastPoint=line[index];
			remain-=d;
			index+=incMode;
		}
	}
	// Add 2 of the endIndex point in to prevent error
	slicedLine.push_back(line[endIndex]);
	slicedLine.push_back(line[endIndex]);
	
	if(incMode==-1){
		for(int i=0;i<n_sq/2;i++){
			Point t=slicedLine[i];
			slicedLine[i]=slicedLine[n_sq-i];
			slicedLine[n_sq-i]=t;
		}
	}
}

void gridify(vector<Point> top,vector<Point> bottom,int page){
	// fill the table for the leftmost row
	cv::line(src_copy, top[0], bottom[0],CV_RGB(255,0,0) , 2, 1,0);
	for(int j=0;j<=n_sq;j++){
		double f=(double)j/n_sq;
		table[page][j][0]=Point(top[0].x+(int)((bottom[0].x-top[0].x)*f),top[0].y+(int)((bottom[0].y-top[0].y)*f));
	}
	
	for(int i=1;i<=n_sq;i++){
		for(int j=0;j<=n_sq;j++){
			double f=(double)j/n_sq;
			int xl=top[i-1].x+(int)((bottom[i-1].x-top[i-1].x)*f);
			int yl=top[i-1].y+(int)((bottom[i-1].y-top[i-1].y)*f);
			int xr=top[i].x+(int)((bottom[i].x-top[i].x)*f);
			int yr=top[i].y+(int)((bottom[i].y-top[i].y)*f);
			table[page][j][i]=Point(xr,yr);

			if(j==0||j==n_sq) cv::line(src_copy, Point(xl,yl), Point(xr,yr),CV_RGB(200,0,0) , 3, 1,0);
			else cv::line(src_copy, Point(xl,yl), Point(xr,yr),CV_RGB(0,200,0) , 3, 1,0);
		}
		if(i==n_sq) cv::line(src_copy, top[i], bottom[i],CV_RGB(255,0,0) , 3, 1,0);
		else cv::line(src_copy, top[i], bottom[i],CV_RGB(0,240,0) , 3, 1,0);
	}
}

void perspectiveTransformation(int page){
    Mat input = src.clone();
	for(int j=1;j<=n_sq;j++){
		for(int i=1;i<=n_sq;i++){
			// Input Quadilateral or Image plane coordinates
			Point2f inputQuad[4]; 
			// Output Quadilateral or World plane coordinates
			Point2f outputQuad[4];
			// Lambda Matrix
			Mat lambda( 2, 4, CV_32FC1 );
			lambda = Mat::zeros(outputSizeH , outputSizeW, input.type() );
			// Output Image;
			Mat output;

			printf("Progress: %.1f Percent \n",((float)(j-1)/n_sq*50+(float)(i-1)/n_sq/n_sq*50)+50*page);
			// The 4 points that select quadilateral on the input , from top-left in clockwise order
			// These four pts are the sides of the rect box used as input 
			inputQuad[0] = Point2f( table[page][j-1][i-1].x	,	table[page][j-1][i-1].y );
			inputQuad[1] = Point2f( table[page][j-1][i].x	,	table[page][j-1][i].y );
			inputQuad[2] = Point2f( table[page][j][i].x		,	table[page][j][i].y );
			inputQuad[3] = Point2f( table[page][j][i-1].x	,	table[page][j][i-1].y );
			// The 4 points where the mapping is to be done , from top-left in clockwise order
			outputQuad[0] = Point2f( 0,0 );
			outputQuad[1] = Point2f( outputSizeW-1,0);
			outputQuad[2] = Point2f( outputSizeW-1,outputSizeH-1);
			outputQuad[3] = Point2f( 0,outputSizeH-1  );
			// Get the Perspective Transform Matrix ie lambda 
			lambda = getPerspectiveTransform( inputQuad, outputQuad );
			Mat inv_lambda(lambda.inv());

			double array_lamb[3][3];
			for(int y=0;y<3;y++){
				for(int x=0;x<3;x++){
					array_lamb[y][x]=inv_lambda.at<double>(y,x);
				}
			}

			for(int y=0;y<outputSizeH;y++){
				for(int x=0;x<outputSizeW;x++){
					float v_x=(float)( x*array_lamb[0][0]	+	y*array_lamb[0][1]	+	array_lamb[0][2]);
					float v_y=(float)( x*array_lamb[1][0]	+	y*array_lamb[1][1]	+	array_lamb[1][2]);
					float v_z=(float)( x*array_lamb[2][0]	+	y*array_lamb[2][1]	+	array_lamb[2][2]);
					map_x.at<float>(outputSizeH*(j-1)+y	,	outputSizeW*(i-1)+outputSizeW*(page*(n_sq))+x)	= v_x/v_z;
					map_y.at<float>(outputSizeH*(j-1)+y	,	outputSizeW*(i-1)+outputSizeW*(page*(n_sq))+x)	= v_y/v_z;
				}
			}
		}
	}
	if(page==1) printf("Progress: COMPLETE\n");
}

/* -----------------------H E L P E R   F U N C T I O N ------------------------ */

double distanceP(int x1,int y1,int x2,int y2){
	return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

Point pointInBetween(int x1,int y1,int x2,int y2,double d){
	double fraction = d/distanceP(x1,y1,x2,y2);
	int x = (int)((x2-x1)*fraction)+x1;
	int y = (int)((y2-y1)*fraction)+y1;
	return Point(x,y);
}

double calcLength(vector<Point> line,int startIndex,int endIndex){
	if(startIndex>endIndex){
		int t=startIndex;
		startIndex=endIndex;
		endIndex=t;
	}
	double length=0;
	for(int i=startIndex;i<=endIndex-1;i++){
		length+=distanceP(line[i].x,line[i].y,line[i+1].x,line[i+1].y);
	}
	return length;
}

void drawLine(Mat m,vector<Point> line,int startIndex,int endIndex, cv::Scalar color, bool drawCircle){
	for(int i=startIndex;i<=endIndex-1;i++){
		cv::line(m, line[i], line[i+1], color, 3, 8,0);
		if(drawCircle){
			cv::circle(m, line[i], 3, CV_RGB(255,0,255), 3, 8,0);
		}
	}
}

/* i= row, j= column */
Point findTLBorder() {
	int rows = dst.rows;
	int column = dst.cols;
	int length=min(rows,column);
	int i=-1,j=-1;
	int flag=0;
	for ( int max = 0; max < length; max++) {
		for (i = 0; i < max; i++ ) {
			j = max-i;
			if((int)dst.at<uchar>(i,j)>threshold_value) {
				flag=1;
				break;
			}
		}
		if(flag==1) break;
	}
	return Point(j,i);
}