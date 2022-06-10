#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "StructurePropagation.h"
#include "OpenCvUtility.h"

using namespace std;
using namespace cv;

#define LINE_STRUCTURE 0
#define CURVE_STRUCTURE 1

#define ks 50
#define ki 2

Mat img;
Mat mask;
Mat mask_inv;
Mat draw_mask;
Mat show_brush;
Mat img_masked;
Mat draw_structure;
Mat mask_structure;
Mat sp_result;
Mat ts_result;
Point pt;
Point prev_pt;
Point points[2] = {Point(-1, -1), Point(-1, -1)};
vector<Point> curvepoints;
vector<vector<Point>> plist;
vector<vector<Point>> mousepoints;
StructurePropagation SP;
int brush_size;
int img_current = 1;
int block_size = 20;
int sample_step = 10;
int line_or_curve = LINE_STRUCTURE;
int points_i=0;

void getImage();
void getMask();
static void callback_draw_mask(int event, int x, int y, int flags, void* param);
void showInterface();
static void callback_draw_line(int event, int x, int y, int flags, void* param);

int main(int argc, char* argv[]) {
    getImage();
    getMask();
    showInterface();
    return 0;
}


void getImage() {
    cout << "Choosing images..." << endl;
    string img_path = "test_data/img/";
    vector<cv::String> files;
    // get the number of images in the directory
    cv::glob(img_path, files, false);
    int img_num = files.size();
    img_current = 0;
    char k = waitKey(0);
    img = imread(img_path + "img" + to_string(img_current) + ".png", 1);
    imshow("img", img);
    // 27: escape key
    while (k != 27) {
        // last image
        if (k == '[') {
            img_current = (img_current + img_num - 1) % img_num;
        }
            // next image
        else if (k == ']') {
            img_current = (img_current + 1) % img_num;
        }
        img = imread(img_path + "img" + to_string(img_current) + ".png", 1);
        imshow("img", img);
        k = waitKey(0);
    }
    destroyAllWindows();
}

void getMask() {
    mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
    draw_mask = img.clone();
    show_brush = draw_mask.clone();
    brush_size = 20;
    prev_pt = Point(-1, -1);

    namedWindow("draw mask");
    imshow("draw mask", show_brush);
    setMouseCallback("draw mask", callback_draw_mask);

    char k = waitKey(0);
    while (k != 27) {
        if (k == '[') {
            if (brush_size > 1) {
                brush_size--;
            }
        }
        // larger brush
        else if (k == ']') {
            if (brush_size < 40) {
                brush_size++;
            }
        }

        // enter key: use default mask
        else if (k == 13) {
            mask = imread("test_data/mask/mask" + to_string(img_current) + ".png", 0);
            break;
        }

        // reset
        else if (k == 'r') {
            mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
            draw_mask = img.clone();
            prev_pt = Point(-1, -1);
        }

        show_brush = draw_mask.clone();
        circle(show_brush, pt, brush_size, Scalar(255, 0, 255), -1);
        imshow("draw mask", show_brush);

        k = waitKey(0);
    }
    threshold(mask, mask_inv, 100, 255, CV_THRESH_BINARY_INV);
    destroyAllWindows();
    img_masked = Mat::zeros(img.size(), CV_8UC3);
    img.copyTo(img_masked, mask_inv);
}


static void callback_draw_mask(int event, int x, int y, int flags, void* param) {
    pt = Point(x, y);
    if ((event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)) || event == CV_EVENT_LBUTTONDOWN) {
        if (prev_pt.x == -1) {
            prev_pt = pt;
        }
        line(mask, prev_pt, pt, Scalar(255), 2 * brush_size);
        line(draw_mask, prev_pt, pt, Scalar(255, 0, 0), 2 * brush_size);
        prev_pt = pt;
    }
    else if (event == CV_EVENT_LBUTTONUP) {
        prev_pt = Point(-1, -1);
    }

    show_brush = draw_mask.clone();
    circle(show_brush, pt, brush_size, Scalar(255, 0, 255), -1);
    imshow("draw mask", show_brush);
}


/**
 * Show the user interface for drawing structural lines and curves.
 * Press Key 's' to run structure propagation, and Key 't' to run texture synthesis.
 * Press Key 'r' to reset, and Key 'a' to save.
 * Press Key 'e' to show curve points.
 */
void showInterface() {
    prev_pt = Point(-1, -1);
    sp_result = img_masked.clone();
    draw_structure = img_masked.clone();
    mask_structure = Mat::zeros(img.rows, img.cols, CV_8UC1);

    ofstream f;
    f.open("point_list/plist" + to_string(img_current) + ".txt", ios::out); // clear old data
    f.close();

    namedWindow("run", 0);
    createTrackbar("Block Size", "run", &block_size, 50);
    createTrackbar("Sample Step", "run", &sample_step, 20);
    createTrackbar("Line or Curve", "run", &line_or_curve, 1);
    imshow("run", draw_structure);
    setMouseCallback("run", callback_draw_line);

    char k = waitKey(0);
    while (k != 27)
    {
        // structure propagation
        Mat mask_structure_tmp = Mat::zeros(img.rows, img.cols, CV_8UC1);
        if (k == 's')
        {
            // run structure propagation
            SP.SetParam(block_size, sample_step, line_or_curve, ks, ki);
            cout << plist.size() << endl;
            SP.Run(mask_inv, img_masked, mask_structure_tmp, plist, sp_result);

            mask_structure_tmp.copyTo(mask_structure, mask_structure_tmp);
            draw_structure = sp_result.clone();
            imshow("run", draw_structure);
            img_masked = sp_result;
            plist.clear();
            mousepoints.clear();
        }
            // reset
        else if (k == 'r')
        {
            draw_structure = img_masked.clone();
            sp_result = img_masked.clone();
            mask_structure = Mat::zeros(img.rows, img.cols, CV_8UC1);
            imshow("run", draw_structure);

            plist.clear();
            mousepoints.clear();
        }
        // save
        else if (k == 'a')
        {
            imwrite("sp_result/sp" + to_string(img_current) + ".png", sp_result); // structure result
            imwrite("ts_result/ts" + to_string(img_current) + ".png", ts_result); // texture result
            imwrite("mask_structure/mask_s" + to_string(img_current) + ".bmp", mask_structure); // structure mask
        }
        // show curve points
        else if (k == 'e')
        {
            plist.resize(mousepoints.size());
            for (int i = 0; i < mousepoints.size(); i++)
            {
                GetCurve(mousepoints[i], plist[i]);
                DrawPoints(plist[i], draw_structure, CV_RGB(0, 0, 255), 1);
            }
            imshow("run", draw_structure);
        }
            // texture synthesis
        else if (k == 't') {
            Mat tmp = sp_result.clone();
            for (int i = 0; i < plist.size(); i++) {
                DrawPoints(plist[i], img, CV_RGB(255, 0, 0), 1);
            }
            Mat tp_result;
            sp_result.copyTo(tp_result);
//            imshow("img", tp_result);
            SP.TextureCompletion(mask, mask_structure, tmp, tp_result);
//            imshow("run", tp_result);
        }

        k = waitKey(0);
    }
}

/**
 * Mouse callback function for drawing the structure lines/curves.
 */
static void callback_draw_line(int event, int x, int y, int flags, void* param) {
    if (line_or_curve == LINE_STRUCTURE) {
        if (event != CV_EVENT_LBUTTONDOWN) {
            return;
        }
        points[points_i].x = x;
        points[points_i].y = y;
        points_i = (points_i + 1) % 2;

        if (points[0].x != -1 && points[1].x != -1 && points_i == 0) {
            vector<Point> line;
            LineInterpolation(points, line);
            plist.push_back(line);

            DrawPoints(line, draw_structure, Scalar(255, 0, 255), 1);
            circle(draw_structure, points[0], 3, Scalar(255,0,0), CV_FILLED);
            circle(draw_structure, points[1], 3, Scalar(255,0,0), CV_FILLED);

            imshow("run", draw_structure);
        }
    }
    // CURVE_STRUCTURE
    else {
        if (event == CV_EVENT_LBUTTONUP) {
            prev_pt = Point(-1, -1);
            mousepoints.push_back(curvepoints);
            vector<Point> tempcurve;
            GetCurve(curvepoints, tempcurve);
            plist.push_back(tempcurve);
            curvepoints = vector<Point>();
        }
        if ( event == CV_EVENT_LBUTTONDOWN ) {
            prev_pt = Point(x, y);
            curvepoints.push_back(prev_pt);
        }
        else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)) {
            pt = Point(x,y);
            curvepoints.push_back(pt);

            if (prev_pt.x < 0) {
                prev_pt = pt;
            }
            line(draw_structure, prev_pt, pt, cvScalarAll(255), 1, 8, 0);
            prev_pt = pt;
            imshow("run", draw_structure);
        }
    }
}