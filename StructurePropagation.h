#ifndef STRUCTUREPROPAGATION_H
#define STRUCTUREPROPAGATION_H
#include<opencv2\opencv.hpp>
#include<vector>
using namespace cv;
using namespace std;

typedef enum {
    INNER, BOUNDARY, OUTER
}AnchorType;

class AnchorPoint {
public:
    int anchor_point;
    AnchorType type;
    int block_size;
    int curve_index;
    std::vector<int> neighbors;

    AnchorPoint(int anchor_point, int block_size, AnchorType type, int curve_index){
        this->anchor_point = anchor_point;
        this->block_size = block_size;
        this->type = type;
        this->curve_index = curve_index;
    }
};


class StructurePropagation {

private:
    int block_size;
    int sample_step;
    int line_or_curve;
    double ks;
    double ki;
    vector<bool>isSingle;
    int max_unknownsize;

    void getAnchorOneCurve(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample, int curve_index);
    Point getLeftTopPoint(Point p);
    Point getPatch(AnchorPoint ap, int curve_index);
    Rect getRect(AnchorPoint ap, int curve_index);
    Mat getOnePatch(Point p,Mat img);
    Mat getOnePatch(AnchorPoint ap, Mat img, int curve_index);
    void copyPatchToImg(AnchorPoint unknown, Mat &patch, Mat &img, int curve_index);
    void getAnchors();
    void drawAnchors();
    Mat PhotometricCorrection(Mat &patch, Mat &mask, Mat &img, Rect &rec);

    bool isBoundary(Point p, bool isSample);
    bool isNeighbor(Point point1, Point point2);
    bool isIntersect(int curve1, int curve2);
    void mergeCurves();

    double calDistance(vector<Point>ci, vector<Point>cxi);
    double calSSD(Mat m1, Mat m2);
    double calEi(AnchorPoint unknown, AnchorPoint sample);
    double calEs(AnchorPoint unknown, AnchorPoint sample);
    double calE1(AnchorPoint unknown, AnchorPoint sample);
    double calE2(AnchorPoint unknown1, AnchorPoint unknown2, AnchorPoint sample1, AnchorPoint sample2);

    vector<int> DP(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample);
    vector<int> BP(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample);

public:
    Mat mask;
    Mat img;
    vector<vector<Point>> pointlist;
    vector<vector<AnchorPoint>> unknown_anchors;
    vector<vector<AnchorPoint>> sample_anchors;
    StructurePropagation() = default;
    void SetParam(int block_size, int sample_step, int line_or_curve, double ks, double ki);
    void Run(const Mat &mask, const Mat& img, Mat &mask_structure, vector<vector<Point>> &plist, Mat& result);
    void TextureCompletion(Mat _mask, Mat LineMask, const Mat &mat, Mat &result);
};

#endif