#include <opencv2/opencv.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/photo.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <chrono>
#include <time.h>
#include <stdio.h>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

void help()
{
	cout << "\nThis program demonstrates line finding with the Hough transform.\n"
		"Usage:\n"
		"./houghlines <image_name>, Default is image.jpg\n" << endl;
}

void detectObject(const std::string& filename) {

	auto start = std::chrono::system_clock::now();
	//auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);

	Mat src = imread(filename, 1); //cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH
	if (src.empty())
	{
		help();
		cout << "can not open " << filename << endl;
		return;
	}

	// This part is for trying Laplace filter and shapening image, then by using
	// Canny edge detector and Hough to get lines, but for now it performs bad
	/*Mat kernel = (Mat_<float>(3, 3) <<
	1, 1, 1,
	1, -8, 1,
	1, 1, 1);*/
	Mat dst, cdst, srcblur, srcblur2, srcmed, dst2, cdst2, srcdilate, imgLaplacian, dstsharp, cdstsharp, srcmedLap, srcDenoised, dst3, cdst3, src_gray, equ, th1, th2, th3;
	//Mat sharp = src; // copy source image to another temporary one
	//filter2D(sharp, imgLaplacian, CV_32F, kernel);
	//src.convertTo(sharp, CV_32F);
	//Mat imgResult = sharp - imgLaplacian;
	//// convert back to 8bits gray scale
	//imgResult.convertTo(imgResult, CV_8UC3);
	//imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	//// imshow( "Laplace Filtered Image", imgLaplacian );
	//imshow("New Sharped Image", imgResult);
	//imshow("Image Laplace", imgLaplacian);

	//fastNlMeansDenoising(imgLaplacian, srcmedLap, 3, 7, 21);
	//imshow("Denoised", srcmedLap);
	////medianBlur(imgLaplacian, srcmedLap, 5);
	//Canny(srcmedLap, dstsharp, 100, 200, 3);
	//cvtColor(dstsharp, cdstsharp, CV_GRAY2BGR);
	//
	//vector<Vec4i> lines;
	//HoughLinesP(dstsharp, lines, 1, CV_PI / 180, 50, 50, 10);
	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	Vec4i l = lines[i];
	//	line(cdstsharp, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, CV_AA);
	//}

	//imshow("detected sharp lines", cdstsharp);
	//waitKey();

	//it is used to remove noise from color images. (Noise is expected to be gaussian)
	// fastNlMeansDenoisingColored(src, None, 10, 10, 7, 21);

	// medianBlur(src, srcmed, 5);
	GaussianBlur(src, srcblur, Size(1, 1), 0, 0, 4);
	//dilate(src, srcdilate, 0, Point(-1,-1), 1, 1, 1);
	/*fastNlMeansDenoising(src, srcDenoised, 3, 7, 21);*/

	// Before Canny threshold(img_tmp,bin_img,30,255,cv::THRESH_BINARY); can be used
	// thresholding the image, cleaning it a bit (using morphology), invert (so that boundaries are white) and send it to Hough
	Canny(srcblur, dst, 190, 200, 3, true);
	cvtColor(dst, cdst, CV_GRAY2BGR);

	/*Canny(srcmed, dst2, 190, 200, 3);
	cvtColor(dst2, cdst2, CV_GRAY2BGR);

	Canny(srcDenoised, dst3, 190, 200, 3);
	cvtColor(dst3, cdst3, CV_GRAY2BGR);*/

	/*vector<Vec4i> lines3;
	HoughLinesP(dst3, lines3, 1, CV_PI / 180, 50, 50, 10);
	for (size_t i = 0; i < lines3.size(); i++)
	{
	Vec4i l = lines3[i];
	line(cdst3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, CV_AA);
	std::cout << lines3[i] << ' ' << '\n';
	}

	imshow("Denoised Detected Lines", cdst3);*/

	/*int aa;
	aa = lines3.size;
	Vec4i le = lines3[0];
	Vec4i lel = lines3[aa];
	line(cdst3, Point(le[0], le[1]), Point(lel[2], lel[3]), Scalar(0, 255, 0), 2, CV_AA);
	imshow("Denoised Detected Lines2", cdst3);*/

	int kernel_size = 1;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_64F; // CV_16S; //CV_8UC1
	char* window_name = "Laplace Demo";
	GaussianBlur(src, srcblur2, Size(7, 7), 0, 0, 4);
	cvtColor(srcblur2, src_gray, CV_BGR2GRAY);



	Laplacian(src_gray, dst2, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(dst2, cdst2);
	cdst2 = cdst2 > 8;
	cvtColor(cdst2, cdst3, CV_GRAY2BGR);
	imshow(window_name, cdst3);
	cout << cdst2.type() << endl;
	//Mat kernel = (Mat_<float>(3, 3) <<
	//	0, 1, 0, //0, -1, 0 for negative
	//	1, -4, 1, // -1, 4, -1
	//	0, 1, 0); // 0, -1, 0
	//char* window_name = "Laplace Demo";
	//GaussianBlur(src, srcblur2, Size(3, 3), 0, 0, 4);
	//cvtColor(srcblur2, src_gray, CV_BGR2GRAY);
	//filter2D(src_gray, imgLaplacian, CV_32F, kernel);
	////cvtColor(imgLaplacian, cdst3, CV_GRAY2BGR);
	//imshow(window_name, imgLaplacian);


	fastNlMeansDenoising(src_gray,srcDenoised, 11.0F, 7, 21);
	imshow("Denoised 1",srcDenoised);

	//equalizeHist(srcDenoised, equ);
	//imshow("Equilize Histogram", equ);

	int thr1 = 400;
	int thr2 = 400;

	for (int i = src.rows ; i < src.rows ; i++)
	{
		for (int j = src.cols ; j < src.cols ; j++)
		{
			if (srcDenoised.at<uchar>(i, j) >= thr1)
			{
				srcDenoised.at<uchar>(i, j) = 65000;
			}
			else
			{
				srcDenoised.at<uchar>(i, j) = 0;
			}
		}
	}

	imshow("Gray Hand Thresholding", srcDenoised);

	/*threshold(src_gray, th1, 400, 65000, THRESH_BINARY);
	imshow("TH1",th1);
	adaptiveThreshold(src_gray, th2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, -7);
	imshow("TH2", th2);
	threshold(src_gray, th3, 400, 65000, THRESH_BINARY+THRESH_OTSU);
	imshow("TH3", th3);*/

	/////// http://stackoverflow.com/questions/11878281/image-sharpening-using-laplacian-filter /////////////

	vector<Vec4i> lines;
	HoughLinesP(src, lines, 1, CV_PI / 180, 60, 50, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, CV_AA);
		std::cout << "Laplacian Lines:" << lines[i] << ' ' << '\n';
    }
	imshow("detected lines 2", src);
	imshow("Gaussian Blurred Demo", srcblur2);
	imshow("Laplacian Demo", cdst2);

#if 0
	vector<Vec2f> lines;
	HoughLines(dst, lines, 1, CV_PI / 180, 100, 0, 0);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	}
#else
	vector<Vec4i> lines2;
	HoughLinesP(dst, lines2, 1, CV_PI / 180, 60, 50, 10);
	for (size_t i = 0; i < lines2.size(); i++)
	{
		Vec4i l = lines2[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, CV_AA);
		std::cout << "Canny Lines:" << lines2[i] << ' ' << '\n';
	}
#endif

	// Another imshow for detected lines
	// Prepare blank mat with same sizes as image
	//Mat Blank(src.rows, src.cols, CV_8UC3, Scalar(0, 0, 0));

	//// Draw lines into image and Blank images
	//for (size_t i = 0; i < lines2.size(); i++)
	//{
	//	Vec4i l = lines2[i];

	//	line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, CV_AA);

	//	line(Blank, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 2, CV_AA);

	//}

	//imwrite("houg.jpg", image);
	//imshow("Edges", src);

	auto end = std::chrono::system_clock::now();
	auto elapsed = end - start;
	std::cout << elapsed.count() << '\n';

	//imshow("source dilate", srcdilate);
	//imshow("source medianblur", srcmed);
	imshow("source", src);
	imshow("source blur", srcblur);
	imshow("source2", dst);
	imshow("detected lines", cdst);
	//imshow("detected lines 2",cdst2);

	//Transformation Part
	//double XM_PI = 3.14159;
	double Azimuth = 24;
	double Altitude = 32;
	double Azimuth_delta = 12;
	double Altitude_delta = 16;
	double Pitch = 100;
	double x = 0, y = 0;
	double x_center = src.rows / 2;
	double y_center = src.cols / 2;
	cout << "Height :" << src.rows << endl;
	cout << "Center in y coordinate :" << src.rows/2 << endl;
	cout << "Width :" << src.cols << endl;
	cout << "Center in x coordinate :" << src.cols/2 << endl;

	double matrix1[2][1] = { { x },
	{ y }};
	double matrix2[2][2] = { { 1/Pitch, 0 },
	{ 0, 1/Pitch } };
	double matrix3[2][1] = { { Azimuth_delta },
	{ Altitude_delta } };
	double matrix4[2][1] = { { x_center },
	{ y_center } };
	double z[2][1];

	int i, j, k;
	for (i = 0; i<2; i++)
		for (j = 0; j<1; j++)
		{
			z[i][j] = 0;
			for (k = 0; k<2; k++)
				// same as z[i][j] = z[i][j] + x[i][k] * y[k][j];
				z[i][j] += matrix2[i][k] * matrix3[k][j];
		}

	/*Adding Two matrices */

	for (i = 0; i<2; ++i)
		for (j = 0; j<1; ++j)
			matrix1[i][j] = matrix4[i][j] + z[i][j];

	cout << endl << "Coordinates in az-el information: " << endl;
	for (i = 0; i<2; ++i)
		for (j = 0; j<1; ++j)
		{
			cout << matrix1[i][j] << "  ";
			if (j == 1 - 1)
				cout << endl;
		}

	// End of transformation part


	/*double clax = 1 * cos((Altitude / 360)*(2 * XM_PI)) * cos((Azimuth / 360)*(2 * XM_PI));
	double clay = 1 * sin((Altitude / 360)*(2 * XM_PI));
	double claz = 1 * cos((Altitude / 360)*(2 * XM_PI)) * sin((Azimuth / 360)*(2 * XM_PI));*/



}

// Read the image
int main(int argc, char** argv)
{
    std::string filename = argc >= 2 ? argv[1] : "./image.jpg";

	detectObject(filename);

	waitKey();

    return 0;
}
