#include"math_function.h"
#include"param.h"
#include<limits>
//double calcuSSD(Mat m1, Mat m2) {
//    if (m1.empty() || m2.empty()) {
//        cout << "In calcuSSD: The mat is empty" << endl;
//        throw exception();
//    }
//    Mat result(1, 1, CV_32F);
//    matchTemplate(m1, m2, result, CV_TM_SQDIFF_NORMED);
//    return result.at<double>(0, 0);
//}

//double calcuDistance(vector<Point>ci, vector<Point>cxi) {
//    double result = 0;
//    double shortest, sq;
//
//    double normalized = norm(Point(PatchSizeCol, PatchSizeRow));
//
//    for (int i = 0; i < ci.size(); i++) {
//        shortest = FLT_MAX;
//        for (int j = 0; j < cxi.size(); j++) {
//            sq = norm(ci[i] - cxi[j]) / normalized;
//            sq *= sq;
//            if (sq < shortest) {
//                shortest = sq;
//            }
//        }
//        result += shortest;
//    }
//    return result;
//}

void initArray(double*a, int num) {
    for (int i = 0; i < num; i++) {
        a[i] = 0;
    }
}
void initArray(int *a, int num) {
    for (int i = 0; i < num; i++) {
        a[i] = 0;
    }
}
void initArray(bool*a, int num) {
    for (int i = 0; i < num; i++) {
        a[i] = false;
    }
}

void addArray(double*a, double*b, double*c, int num) {
    for (int i = 0; i < num; i++) {
        a[i] = a[i] + b[i];
    }
}
void minusArray(double*a, double*b, double *c, int num) {
    for (int i = 0; i < num; i++) {
        a[i] = a[i] - b[i];
    }
}
bool isEqualArray(double *a, double*b, int num) {
    for (int i = 0; i < num; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}
void moveArray(double*a, double*b, int num) {
    for (int i = 0; i < num; i++) {
        a[i] = b[i];
    }
}
bool contain(Rect &rec, Point &p) {
    if (p.x >= rec.x&&p.x <= rec.x + rec.width && p.y >= rec.y&&p.y <= rec.y + rec.height) {
        return true;
    }
    return false;
}
