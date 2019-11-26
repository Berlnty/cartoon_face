#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <iostream>
#include <stdio.h>
#include <iomanip> 
#include<cmath>
#include <math.h> 


using namespace std;
using namespace cv;

#define PI 3.14159265
RNG rng(12345);
Point first, second;


void removePepperNoise(Mat &mask)
{
	// For simplicity, ignore the top & bottom row border.
	for (int y = 2; y<mask.rows - 2; y++) {
		// Get access to each of the 5 rows near this pixel.
		uchar *pThis = mask.ptr(y);
		uchar *pUp1 = mask.ptr(y - 1);
		uchar *pUp2 = mask.ptr(y - 2);
		uchar *pDown1 = mask.ptr(y + 1);
		uchar *pDown2 = mask.ptr(y + 2);

		// For simplicity, ignore the left & right row border.
		pThis += 2;
		pUp1 += 2;
		pUp2 += 2;
		pDown1 += 2;
		pDown2 += 2;
		for (int x = 2; x<mask.cols - 2; x++) {
			uchar v = *pThis;   // Get the current pixel value (either 0 or 255).
								// If the current pixel is black, but all the pixels on the 2-pixel-radius-border are white
								// (ie: it is a small island of black pixels, surrounded by white), then delete that island.
			if (v == 0) {
				bool allAbove = *(pUp2 - 2) && *(pUp2 - 1) && *(pUp2) && *(pUp2 + 1) && *(pUp2 + 2);
				bool allLeft = *(pUp1 - 2) && *(pThis - 2) && *(pDown1 - 2);
				bool allBelow = *(pDown2 - 2) && *(pDown2 - 1) && *(pDown2) && *(pDown2 + 1) && *(pDown2 + 2);
				bool allRight = *(pUp1 + 2) && *(pThis + 2) && *(pDown1 + 2);
				bool surroundings = allAbove && allLeft && allBelow && allRight;
				if (surroundings == true) {
					// Fill the whole 5x5 block as white. Since we know the 5x5 borders
					// are already white, just need to fill the 3x3 inner region.
					*(pUp1 - 1) = 255;
					*(pUp1 + 0) = 255;
					*(pUp1 + 1) = 255;
					*(pThis - 1) = 255;
					*(pThis + 0) = 255;
					*(pThis + 1) = 255;
					*(pDown1 - 1) = 255;
					*(pDown1 + 0) = 255;
					*(pDown1 + 1) = 255;
				}
				// Since we just covered the whole 5x5 block with white, we know the next 2 pixels
				// won't be black, so skip the next 2 pixels on the right.
				pThis += 2;
				pUp1 += 2;
				pUp2 += 2;
				pDown1 += 2;
				pDown2 += 2;
			}
			// Move to the next pixel.
			pThis++;
			pUp1++;
			pUp2++;
			pDown1++;
			pDown2++;
		}
	}
}

void cartoonifyImage(Mat srcColor, Mat &dst)
{
	// Convert from BGR color to Grayscale
	Mat srcGray;
	cvtColor(srcColor, srcGray, CV_BGR2GRAY);

	// Remove the pixel noise with  Median filter, before we start detecting edges.
	medianBlur(srcGray, srcGray, 7);

	Size size = srcColor.size();
	Mat mask = Mat(size, CV_8U);
	Mat edges = Mat(size, CV_8U);

	// Generate a nice edge mask, similar to a pencil line drawing.
	Laplacian(srcGray, edges, CV_8U, 5);
	threshold(edges, mask, 140, 255, THRESH_BINARY_INV);
	// Mobile cameras usually have lots of noise, so remove small
	// dots of black noise from the black & white edge mask.
	removePepperNoise(mask);
	bitwise_not(mask, dst);


}

int contors(Mat image, string s)
{



	// from rgb to hsv
	Mat hsv_image;
	cvtColor(image, hsv_image, CV_BGR2HSV);

	
	// thresholding
	//Scalar(0, 48, 8), Scalar(20, 255, 255) 46
	//Scalar(0, 50, 0), Scalar(120, 150, 255)
	// Scalar(0, 133, 77), Scalar(255, 173, 127)
	Mat skinmask;

	inRange(hsv_image, Scalar(0, 48, 8), Scalar(20, 255, 255), skinmask);

	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(11, 11), Point(0, 0));
	erode(skinmask, skinmask, kernel);
	dilate(skinmask, skinmask, kernel);

	GaussianBlur(hsv_image, hsv_image, Size(3, 3), 0);


	Mat skin;
	bitwise_and(image, image, skin, skinmask);
	//imshow(s + "original", image);
	imshow(s + "mask", skinmask);
	imshow(s + "skin", skin);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(skinmask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<RotatedRect>  minRec(contours.size());
	vector<RotatedRect> minEllipse(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		minRec[i] = minAreaRect(Mat(contours[i]));
		
	}

	Mat drawing = Mat::zeros(skin.size(), CV_8UC3);
	vector<pair<int, int>> areas;
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());

		
		// rotated rectangle
		Point2f rect_points[4];
		minRec[i].points(rect_points);
		pair<int, int> dump;
			int area = sqrt(pow(rect_points[1].x - rect_points[0].x, 2) + pow(rect_points[1].y - rect_points[1].y, 2)) * sqrt(pow(rect_points[3].x - rect_points[2].x, 2) + pow(rect_points[3].y - rect_points[2].y, 2));
			cout << endl << "area: " << area << "  " << 0.5*drawing.cols*drawing.rows << endl;
			if (area < 0.5*drawing.cols*drawing.rows) {
				dump.first = area; dump.second = i;
				areas.push_back(dump);
			}
			if (true) {
			for (int j = 0; j < 4; j++)
				line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
		}
	}
	cout <<endl<< "data : " << s << ' ' << contours.size() << " " << areas.size() << endl;

	first = Point(-1,-1);
	second = Point(-1, -1);
	if (areas.size() >= 2) {
		sort(areas.begin(), areas.end());
		pair<int, int>maximumarea =areas[areas.size()-1];
		first.x = contours[maximumarea.second][0].x;
		first.y = contours[maximumarea.second][0].y;
		for (int i = 1; i < contours[maximumarea.second].size(); ++i)
		{
			first.x = min(first.x, contours[maximumarea.second][i].x);
			first.y = max(first.y, contours[maximumarea.second][i].y);

	}
		maximumarea = areas[areas.size() - 2];
		second.x = contours[maximumarea.second][0].x;
		second.y = contours[maximumarea.second][0].y;
		for (int i = 1; i < contours[maximumarea.second].size(); ++i)
		{
			second.x = min(second.x, contours[maximumarea.second][i].x);
			second.y = max(second.y, contours[maximumarea.second][i].y);

		}
		
	}
	else if (areas.size() == 1) {
		first.x = contours[areas[0].second][0].x;
		first.y = contours[areas[0].second][0].y;

		second = first;
	}
	imshow(s + "Contours", drawing);

	return 0;
}

int cpyeyes(Mat image, Mat eyes, int startx, int starty)
{

	cvtColor(eyes, eyes, CV_BGR2BGRA);

	for (int i = 0; i < eyes.rows; i++)
	{
		for (int j = 0; j < eyes.cols; j++)
		{
			if ((int)eyes.at<cv::Vec4b>(i, j)[3] != 0)
			{
				image.at<cv::Vec3b>(i + starty, j + startx)[0] = eyes.at<cv::Vec4b>(i, j)[0];
				image.at<cv::Vec3b>(i + starty, j + startx)[1] = eyes.at<cv::Vec4b>(i, j)[1];
				image.at<cv::Vec3b>(i + starty, j + startx)[2] = eyes.at<cv::Vec4b>(i, j)[2];

			}
		}
	}

	return 0;
}


Mat rotate(Mat src, double angle)
{
	Mat dst;
	Point2f pt(src.cols / 2., src.rows / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
	return dst;
}
// --------------------------------------------------- Generate a cutted image contains eyes -----------------------------------//

Mat cut_eyes(Mat image, Point center, Rect R) {

	static int y = 0;
	int cols = image.cols;
	int rows = image.rows;

	Mat cutted_image;

	// ------------------------------------ only the way to detect the axes of eyes --------------------//
	//--------------------------    x , y , width , height
	Rect cropped_Rectangle = Rect(abs(center.x - (R.width / 2)), abs(center.y - (R.height / 6)), (R.width), R.height / 7);
	Mat CroppedImage = image(cropped_Rectangle);
	imshow(y + "cropped image", CroppedImage);


	y++;
	return CroppedImage;
}
Mat cut_mouth(Mat image, Point center, Rect R) {

	static int y = 0;
	int cols = image.cols;
	int rows = image.rows;

	Mat cutted_image;

	// ------------------------------------ only the way to detect the axes of eyes --------------------//
	//--------------------------    x , y , width , height
	Rect cropped_Rectangle = Rect(abs(center.x - (R.width / 2)), abs(center.y + (R.height / 5)), (R.width), R.height / 7);
	Mat CroppedImage = image(cropped_Rectangle);
	imshow(y + "cropped image", CroppedImage);



	
	y++;
	return CroppedImage;
}
Mat clustring(Mat image)
{


	Mat gray_image, hist;
	cvtColor(image, gray_image, CV_BGR2GRAY);

	int histSize = 256;
	// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	/// Compute the histograms:
	calcHist(&gray_image, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	int sum = 0;
	int n = image.rows * image.cols;

	for (int j = 0; j < 256; j++)
	{
		if (hist.at<float>(j, 0)>0.0085*n)
			sum++;
	}

	int K = min(12, sum);


	Mat data = image.reshape(1, n);
	data.convertTo(data, CV_32F);

	vector<int> labels;
	Mat1f colors;
	kmeans(data, K, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, colors);

	for (int i = 0; i < n; ++i)
	{
		data.at<float>(i, 0) = colors(labels[i], 0);
		data.at<float>(i, 1) = colors(labels[i], 1);
		data.at<float>(i, 2) = colors(labels[i], 2);
	}

	Mat reduced = data.reshape(3, image.rows);
	reduced.convertTo(reduced, CV_8U);


	//	imshow("Reduced", reduced);

	return reduced;

}




 void detect_eyes(Mat image, Point center) {
	Mat gra, hsv_image;

	// convert image to gray scale
	cvtColor(image, gra, CV_RGB2GRAY);

	cvtColor(image, hsv_image, CV_RGB2GRAY);
	imshow("ga", gra);
	int channels = hsv_image.channels(); // number of channels
	int rows = hsv_image.rows;
	int cols = hsv_image.cols* channels;

	// ------------------------------------gray vesrsion--------------------------------------//
	Scalar intensity1 = gra.at<float>(center.x, center.y);

	for (int i = 0; i < image.rows; i++)
	{

		for (int j = 0; j < image.cols; j++) {

			Vec3b hsv = hsv_image.at<cv::Vec3b>(i, j);

			Scalar intensity = gra.at<uchar>(i, j);
			

			// -------------------------------------------detect eyes with gray level-------------------------------------------//

			if ((intensity1.val[0]>intensity.val[0]) && (pow(pow(j - center.x, 2) + pow(i - center.y, 2), 0.5) <= 100)) {
				gra.at<float>(i, j) = 255;
			}
		}

	}
	imshow("graylevel", gra);
}

void detectface(Mat image, string s, Point center, Rect R);
Mat detect_eyes_geometrically(Mat image, Point center, Rect R);

int main()
{
	Mat image;
	image = imread("sis.jpg", CV_LOAD_IMAGE_COLOR);


	//resize(image, image, cv::Size(), 0.4, 0.4);
	// Load Face cascade (.xml file)
	CascadeClassifier face_cascade;
	face_cascade.load("C:/OpenCV3/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml");

	// Detect faces
	std::vector<Rect> faces;
	std::vector<Mat> faceimages;
	face_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(90, 90));

	// Draw circles on the detected faces
	for (int i = 0; i < faces.size(); i++)
	{
		cv::Mat faceb = Mat::zeros(faces[i].height, faces[i].width, CV_8UC3);

		faceb.rows = faces[i].height*0.3;
		faceb.cols = faces[i].width*0.3;
		
		faceb = image(Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height));
	
		faceimages.push_back(faceb);
	
	//	imshow(to_string(i)+"....", faceb);


		
		
	}

	CascadeClassifier eye_cascade;
	eye_cascade.load("C:/OpenCV3/opencv-master/data/haarcascades/haarcascade_eye.xml");


	Mat eyes;
	Mat mouth;
	mouth = imread("3.png", CV_LOAD_IMAGE_UNCHANGED);
	eyes = imread("blueeyes.png", CV_LOAD_IMAGE_UNCHANGED);
	int diagonal = sqrt(pow(eyes.rows, 2) + pow(eyes.cols, 2));
	float eyexratio = (float)eyes.cols / diagonal;
	float eyeyratio = (float)eyes.rows / diagonal;
	cout << "ratios: " << eyexratio << "	" << eyeyratio << endl;

	Mat dump;

	int starty = 0.5*diagonal - 0.5*eyes.rows;
	int startx = 0.5*diagonal - 0.5*eyes.cols;
	resize(eyes, dump, cv::Size(), (float)diagonal / eyes.cols, (float)diagonal / eyes.rows);


	cout << dump.rows << " " << dump.cols;




	for (int i = 0; i < dump.rows; i++)
	{
		for (int j = 0; j < dump.cols; j++)
		{
			if ((int)dump.at<cv::Vec4b>(i, j)[3] != 0)
				dump.at<cv::Vec4b>(i, j)[3] = 0;


		}
	}
	cvtColor(dump, dump, CV_BGR2BGRA);
	for (int i = 0; i < eyes.rows; i++)
	{
		for (int j = 0; j < eyes.cols; j++)
		{

			dump.at<cv::Vec4b>(i + starty, j + startx)[0] = eyes.at<cv::Vec4b>(i, j)[0];
			dump.at<cv::Vec4b>(i + starty, j + startx)[1] = eyes.at<cv::Vec4b>(i, j)[1];
			dump.at<cv::Vec4b>(i + starty, j + startx)[2] = eyes.at<cv::Vec4b>(i, j)[2];
			dump.at<cv::Vec4b>(i + starty, j + startx)[3] = eyes.at<cv::Vec4b>(i, j)[3];

		}
	}



	////---------------------------------------looping on faces to detect eyes----------------------------------

	for (int y = 0; y < faceimages.size(); ++y) {

		Point center(faces[y].x + faces[y].width*0.5, faces[y].y + faces[y].height*0.5);
		detectface(image, to_string(y), center, faces[y]);
		if (first .x == -1 || second.x == -1) continue;
		cout << "first x: " << first.x << "y: " << first.y << endl;
		cout << "second x: " << second.x << "y: " << second.y << endl;

		
		double slope;
		if (first.x>second.x)
			slope = (double)(first.y - second.y) / (first.x - second.x);
		else slope = (double)(second.y - first.y) / (second.x - first.x);

		double theta = atan(slope) * 180 / PI; ;
//		cout << "slope" + to_string(y) + "  : " << eyevec[1].y << "    " << eyevec[0].y << "    " << eyevec[1].x << "   " << eyevec[0].x << " k " << std::setprecision(5) << theta << endl;

		Mat dump3;
		resize(dump, dump3, cv::Size(), float(faces[y].width)*0.8 / eyes.cols, float(faces[y].height)*0.3 / eyes.rows);
		resize(mouth, mouth, cv::Size(), float(faces[y].width)*0.6 / mouth.cols, float(faces[y].height)*0.25 / mouth.rows);
		dump3 = rotate(dump3, -theta);
		Mat dump2;
		dump2 = rotate(mouth, -theta);
		cv::Size size = dump3.size();
		cv::Point startPosition = cv::Point(faces[y].x + 0.4*faces[y].width, faces[y].y + 0.3*faces[y].height);
		Point eyexy;
		if (first.x< second.x)
		{
			eyexy.x = first.x;
			eyexy.y =first.y;
		}
		else {
			eyexy.x = second.x;
			eyexy.y = second.y;
		}
		diagonal = sqrt(pow(dump3.rows, 2) + pow(dump3.cols, 2));
		cout << endl << faces[y].y + first.y << "   " << (0.5*(1 - eyexratio)*diagonal) << "  " << faces[y].x + eyexy.x - (0.5*(1 - eyexratio)*diagonal);
		cout << endl << faces[y].y + eyexy.y << "   " << (0.5*(1 - eyeyratio)*diagonal) << "  " << faces[y].y + eyexy.y - (0.5*(1 - eyeyratio)*diagonal);

		cpyeyes(image, dump3, faces[y].x+ eyexy.x - (0.9*(1 - eyexratio)*diagonal), faces[y].y + eyexy.y);// - (0.3*(1 - eyeyratio)*diagonal));
		float dx, dy;//eyes
		dx = 0.2*faces[y].width;
		dy = (0.72)*faces[y].height;
		double theta2 = atan(slope);
		cpyeyes(image, dump2, faces[y].x + (dx*(cos((PI/2)-theta2)+sin((PI/2)-theta2)) ), faces[y].y + (dy*(cos((PI/2) - theta2) + sin((PI/2) - theta2))));					//mouth
		int borm = 0;
		
	}

	Mat cartoon = image;
	cartoonifyImage(image, cartoon);
	image = clustring(image);

	bitwise_xor(image, image, image, cartoon);
	imshow("final", image);
	
	waitKey(0);
	return 0;
}

void detectface(Mat image, string s, Point center, Rect myimage)
{

	// from rgb to hsv
	Mat hsv_image, gra;
	cvtColor(image, hsv_image, CV_RGB2HSV);
	Mat returned_cut = detect_eyes_geometrically(image, center, myimage);
	contors(returned_cut, s);

}



Mat detect_eyes_geometrically(Mat image, Point center, Rect  R) {
	static char i = 1;
	// convert rbg to gray level
	Mat gra;
	cvtColor(image, gra, CV_RGB2GRAY);

	int cols = image.cols;
	int rows = image.rows;
	Mat returned_cut = cut_eyes(image, center, R);

	Scalar intensity = gra.at<char>(center.x, center.y);


	//eyes here suppose we have a y axis  drawing it  rows == y == i   / cols == x == j
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Scalar intensity1 = gra.at<char>(i, j);

			// draws x axes
			if (i == center.y) {
				gra.at<char>(i, j) = 255;
			}

			// draws y axes
			if (j == center.x) {
				gra.at<char>(i, j) = 255;
			}
		}
	}


	
	// -------------------------------------------- Detecting x  Axes of Eyes ------------------------------------------------- //
	for (int i = center.x - 50; i < center.x + 50; i++) {
		gra.at<char>(center.y - R.height*0.1, i) = 255;
		gra.at<char>(center.y - R.height*0.1, i) = 255;
		gra.at<char>(center.y - R.height*0.1, i) = 255;
	}


	i++;
	return returned_cut;
}


