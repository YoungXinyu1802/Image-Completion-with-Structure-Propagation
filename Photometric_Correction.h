#pragma once
#include<opencv2\opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

class Photometric_Correction {
private:
    Mat mask;
public:
    Photometric_Correction() = default;
    Mat correct(Mat &patch, Mat &imgmask, Mat &resImg,Rect &rec);
};