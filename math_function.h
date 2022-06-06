#pragma once
#include<opencv2\opencv.hpp>
#include<iostream>
//#include"param.h"
#include<vector>
using namespace cv;
using namespace std;

double calcuSSD(Mat m1, Mat m2);
double calcuDistance(vector<Point>ci, vector<Point>cxi);
void initArray(double*a, int num);
void initArray(int *a, int num);
void initArray(bool*a, int num);
void addArray(double*a, double*b, double*c,int num);
void minusArray(double*a, double*b, double *c,int num);
bool isEqualArray(double *a, double*b, int num);
void moveArray(double*a, double*b, int num);
bool contain(Rect &rec, Point &p);
string int_to_string(int i);