#ifndef STRUCTUREPROPAGATION_H
#define STRUCTUREPROPAGATION_H
#include"AnchorPoint.h"
//#include "Photometric.h"
#include"Photometric_Correction.h"
#include<opencv2\opencv.hpp>
#include<vector>
using namespace cv;
using namespace std;

class StructurePropagation {

private:
    int block_size;
    int sample_step;
    int line_or_curve;
    double ks;
    double ki;

    string path = "test_data/result/";
    /*
    the way to find the anchor point is that,from the first point on the curve,each turn we get the half number
    of points of the patch.
    */
    double calDistance(vector<Point>ci, vector<Point>cxi);
    double calSSD(Mat m1, Mat m2);

    int getOneAnchorPos(int lastanchor_index,PointType &t,int curve_index,bool flag, vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample);
    void getOneCurveAnchors(int curve_index,vector<AnchorPoint>&unknown,vector<AnchorPoint>&sample);

    int getOneAnchorFront(int lastanchor_index, PointType &t, int curve_index, bool flag, vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample);
    int getOneAnchorBack(int lastanchor_index, PointType &t, int curve_index, bool flag, vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample);


    Point getAnchorPoint(AnchorPoint ap, int curve_index);
    Rect getRect(AnchorPoint ap, int curve_index);

    Mat getOnePatch(Point p,Mat &img);
    Mat getOnePatch(AnchorPoint ap, Mat &img, int curve_index);
    void copyPatchToImg(AnchorPoint unknown, Mat &patch, Mat &img, int curve_index);

    double calEi(AnchorPoint unknown, AnchorPoint sample, int curve_index);
    double calEs(AnchorPoint unknown, AnchorPoint sample, int curve_index);
    double calE1(AnchorPoint unknown, AnchorPoint sample, int curve_index);
    double calE2(AnchorPoint unknown1, AnchorPoint unknown2, AnchorPoint sample1, AnchorPoint sample2, int curve_index);

    vector<int> DP(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample, int curve_index);
    vector<int> BP(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample, int curve_index);
    //to judge if two points are neighbor
    bool isNeighbor(Point point1, Point point2);
    bool isIntersect(int curve1, int curve2);
    //add the front and the behind anchor point as the neighbor
    void addNeighborFB(int curve_index);
    //to find the intersecting curves and merge them into the first curve
    void mergeCurves(vector<bool>&isSingle);

    //need to be correct ,not done
    void getOneNewCurve(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample, int curve_index, bool flag, Mat &result);



public:
//    Image image;
    Mat mask;
    Mat img;
    vector<vector<Point>> pointlist;
    vector<vector<AnchorPoint>> unknown_anchors;
    vector<vector<AnchorPoint>> sample_anchors;
    Photometric_Correction *pc;
    StructurePropagation() = default;
    void SetParam(int block_size, int sample_step, int line_or_curve, double ks, double ki);
    void Run(const Mat &mask, const Mat& img, Mat &mask_structure, vector<vector<Point>> &plist, Mat& result);
    void TextureCompletion(Mat _mask, Mat LineMask, const Mat &mat, Mat &result);
    void getAnchors();
    void drawAnchors();

    //tool function
    Point getLeftTopPoint(int point_index, int curve_index);
    Point getLeftTopPoint(Point p);
    Rect getRect(Point p);
};

#endif