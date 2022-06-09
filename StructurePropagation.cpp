#include"StructurePropagation.h"
#include"math_function.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
using namespace Eigen;

void StructurePropagation::SetParam(int block_size, int sample_step, int line_or_curve, double ks, double ki) {
    this->block_size = block_size;
    this->sample_step = sample_step;
    this->line_or_curve = line_or_curve;
    this->ks = ks;
    this->ki = ki;
}

double StructurePropagation::calSSD(Mat m1, Mat m2) {
    if (m1.empty() || m2.empty()) {
        cout << "In calSSD: The mat is empty" << endl;
        throw exception();
    }
    Mat result(1, 1, CV_32F);

    // calculate the ssd of the overlap area
    matchTemplate(m1, m2, result, CV_TM_SQDIFF_NORMED);
    return result.at<double>(0, 0);
}

double StructurePropagation::calDistance(vector<Point>ci, vector<Point>cxi) {
    double dist = 0;
    double shortest, s;

    double normalized = norm(Point(block_size, block_size));

    // calculate the shortest distance between ci and cxi
    for (int i = 0; i < ci.size(); i++) {
        shortest = DBL_MAX;
        for (int j = 0; j < cxi.size(); j++) {
            s = norm(ci[i] - cxi[j]) / normalized;
            s *= s;
            shortest = MIN(s, shortest);
        }
        dist += shortest;
    }
    return dist;
}

double StructurePropagation::calEs(AnchorPoint unknown, AnchorPoint sample) {
    vector<Point>ci, cxi;
    //the left_top point of the patches
//    if (unknown.curve_index != sample.curve_index) {
//        cout << "calEs error: not in the same curve" << endl;
//        throw exception();
//    }
    int curve_index = unknown.curve_index;
    Point p1 = pointlist[curve_index][unknown.anchor_point];
    Point p2 = pointlist[curve_index][sample.anchor_point];
    Point origin_unknown = getLeftTopPoint(p1);
    Point origin_sample = getLeftTopPoint(p2);

    //to callate the relative coordinate of the point in the patch
    for (int i = unknown.begin_point; i <= unknown.end_point; i++) {
        Point p = pointlist[curve_index][i];
        ci.push_back(p - origin_unknown);
    }
    for (int i = sample.begin_point; i <= sample.end_point; i++) {
        Point p = pointlist[curve_index][i];
        cxi.push_back(p - origin_sample);
    }

    int num_ci = unknown.end_point - unknown.begin_point + 1;
    int num_cxi = sample.end_point - sample.begin_point + 1;
    double result = calDistance(ci, cxi)/num_ci + calDistance(cxi, ci)/num_ci;

    return result;
}

double StructurePropagation::calEi(AnchorPoint unknown, AnchorPoint sample) {
//    if (unknown.curve_index != sample.curve_index) {
//        cout << "calEi error: not in the same curve" << endl;
//        throw exception();
//    }

    if (unknown.type != BORDER)
        return 0;

    int curve_index = unknown.curve_index;
    Mat patch_image, patch_mask;
    Point p = pointlist[curve_index][unknown.anchor_point];
    getOnePatch(p, img).copyTo(patch_image);
    getOnePatch(p, mask).copyTo(patch_mask);

    Mat unknown_mask(block_size, block_size, CV_8U);
    unknown_mask.setTo(0);
    Mat sample_mask(block_size, block_size, CV_8U);
    sample_mask.setTo(0);
    patch_image.copyTo(unknown_mask, patch_mask);
    Point sp = pointlist[curve_index][sample.anchor_point];
    getOnePatch(sp, img).copyTo(sample_mask, patch_mask);

    return calSSD(unknown_mask, sample_mask);
}

double StructurePropagation::calE1(AnchorPoint unknown, AnchorPoint sample) {
    double  Es = calEs(unknown, sample);
    double  Ei = calEi(unknown, sample);
    double E1 = ks * Es + ki * Ei;
    return E1;
}


double StructurePropagation::calE2(AnchorPoint unknown1, AnchorPoint unknown2, AnchorPoint sample1, AnchorPoint sample2) {
    //get the four vertexes of the two patches
//    if (unknown1.curve_index != sample1.curve_index) {
//        cout << "calE2 error: not in the same curve" << endl;
//        throw exception();
//    }

    int curve_index = unknown1.curve_index;

    Point u1_point = pointlist[curve_index][unknown1.anchor_point];
    Point u1_lefttop = getLeftTopPoint(u1_point);
    Rect rec1(u1_lefttop.x, u1_lefttop.y, unknown1.block_size, unknown1.block_size);

    Point u2_point = pointlist[curve_index][unknown2.anchor_point];
    Point u2_lefttop = getLeftTopPoint(u2_point);
    Rect rec2(u2_lefttop.x, u2_lefttop.y, unknown2.block_size, unknown2.block_size);

    Rect intersect = rec1 & rec2;

    Mat patch1 = getOnePatch(getAnchorPoint(sample1, curve_index), img);
    Mat patch2 = getOnePatch(getAnchorPoint(sample2, curve_index), img);

    Mat copy1 = img.clone();
    Mat copy2 = img.clone();
    //overlap the src image with the corresponding sample patch
    patch1.copyTo(copy1(rec1));
    patch2.copyTo(copy2(rec2));

    double result;
    // calculate the SSD of the overlap parts of two sample patches
    result = calSSD(copy1(intersect), copy2(intersect));

    return result;
}


vector<int> StructurePropagation::DP(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample) {
    int unknown_size = unknown.size();
    int sample_size = sample.size();

    if (unknown_size == 0 || sample_size == 0) {
        cout << "In DP: the size of vector AnchorPoint is 0" << endl;
        throw exception();
    }

    // M(i) = E1 + min{E2 + M(i-1)}
    Eigen::MatrixXd M(unknown_size, sample_size);
    M.fill(0);
    Eigen::MatrixXd record(unknown_size, sample_size);
    record.fill(0);

    // initialize M[0]: M[0][xi] = E[0][xi]
    double E1, E2;
    for (int i = 0; i < sample_size; i++) {
        E1 = calE1(unknown[0], sample[i]);
        M(0, i) = E1;
    }

    // compute M[i][j]
    for (int i = 1; i < unknown_size; i++) {
        for (int xi = 0; xi < sample_size; xi++) {
            double min = DBL_MAX;
            int min_idx = 0;
            E1 = calE1(unknown[i], sample[xi]);
            // min_(x(i-1)){E2(x(i-1), x(i)), M(i-1)(x(i-1))}
            // find the sample index to make it the minimum
            for (int m = 0; m < sample_size; m++) {
                E2 = calE2(unknown[i - 1], unknown[i], sample[m], sample[xi]);
                double tmp = E2 + M(i - 1, m);
                if (tmp < min) {
                    min = tmp;
                    min_idx = m;
                }
            }
            M(i, xi) = E1 + min;
            record(i, xi) = min_idx;
        }
    }
    vector<int> sample_label;
    // find the best patch for the last unknown anchor point
    int last_patch = 0;
    double min = M(unknown_size - 1, 0);
    for (int i = 0; i < sample_size; i++) {
        if (M(unknown_size - 1, i) < min) {
            last_patch = i;
            min = M(unknown_size - 1, i);
        }
    }
    sample_label.insert(sample_label.begin(), last_patch);
    //back tracing
    if (unknown_size > 1) {
        for (int i = unknown_size - 1; i > 0; i--) {
            last_patch = record(i, last_patch);
            sample_label.insert(sample_label.begin(), last_patch);
        }
    }

    return sample_label;
}


bool StructurePropagation::isNeighbor(Point p1, Point p2) {
    int x = abs(p1.x - p2.x);
    int y = abs(p1.y - p2.y);
    return x < block_size && y < block_size;
}

bool StructurePropagation::isIntersect(int curve1, int curve2) {
    if (unknown_anchors[curve1].empty() || unknown_anchors[curve2].empty()) {
        return false;
    }
    int num_curve1 = unknown_anchors[curve1].size();
    int num_curve2 = unknown_anchors[curve2].size();

    for (int i = 0; i < num_curve1; i++) {
        int p1 = unknown_anchors[curve1][i].anchor_point;
        Point point1 = pointlist[curve1][p1];
        for (int j = 0; j < num_curve2; j++) {
            int p2 = unknown_anchors[curve2][j].anchor_point;
            Point point2 = pointlist[curve2][p2];
            if (isNeighbor(point1, point2)) {
                unknown_anchors[curve1][i].neighbors.push_back(j + num_curve1);
                unknown_anchors[curve1][i].neighbors.push_back(j + 1 + num_curve1);
                /*this means that the anchor point i will become the milddle point between j and j+1, so there will no path from j
                to j+1 directly*/
                if (j + 1 < unknown_anchors[curve2].size()) {
                    unknown_anchors[curve2][j].neighbors[1] = i - num_curve1;
                    unknown_anchors[curve2][j + 1].neighbors[0] = i - num_curve1;
                }
                else {
                    unknown_anchors[curve2][j].neighbors.push_back(i - num_curve1);
                }

                return true;
            }
        }
    }
    return false;
}

vector<int> StructurePropagation::BP(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample, int curve_index) {
    cout << "begin to BP ... ..." << endl;

    vector<int>sample_label;
    int unknown_size = unknown.size();
    int sample_size = sample.size();

    Matrix<VectorX<double>, Dynamic, Dynamic> M;
    M.resize(unknown_size, unknown_size);
    VectorX<double> M_in(sample_size);
    M_in.fill(0);
    M.fill(M_in);
    VectorX<VectorX<double>> E1(unknown_size);
    VectorX<double> E1_in(sample_size);
    E1_in.fill(0);
    E1.fill(E1_in);

    // calculate the E1 matrix
    for (int i = 0; i < unknown_size; i++) {
        for (int xi = 0; xi < sample_size; xi++) {
            E1[i][xi] = calE1(unknown[i], sample[xi]);
        }
    }

    //to judge if the node has been converged
    Matrix<bool, Dynamic, Dynamic> isConverge;
    isConverge.resize(unknown_size, unknown_size);
    isConverge.fill(false);

    VectorX<double> sum_vec(sample_size);
    VectorX<double> E_M_sum(sample_size);
    VectorX<double> new_vec(sample_size);

    //begin to iterate
    cout << "unknown_size:" << max_unknownsize << endl;
    for (int t = 0; t < max_unknownsize; t++) {
        cout << "t: " << t << endl;
        for (int node = 0; node < unknown_size; node++) {
            //calculate the sum of M[t-1][i][j]
            sum_vec.fill(0);
            for (int neighbor_index = 0; neighbor_index < unknown[node].neighbors.size(); neighbor_index++) {
                //neighbors to node
                sum_vec = sum_vec - M(neighbor_index, node);
            }
            //node to neighbors
            int size = unknown[node].neighbors.size();
            for (int times = 0; times < unknown[node].neighbors.size(); times++) {
                int neighbor_index = unknown[node].neighbors[times];
                if (isConverge(node, neighbor_index)) {
                    continue;
                }
                sum_vec = sum_vec - M(neighbor_index, node);
                E_M_sum = E_M_sum + E1[node] + sum_vec;

                for (int xj = 0; xj < sample_size; xj++) {
                    double min = DBL_MAX;
                    for (int xi = 0; xi < sample_size; xi++) {
                        double E2 = calE2(unknown[node], unknown[neighbor_index], sample[xi], sample[xj]);
                        double sum = E2 + E_M_sum[xi];
                        if (sum < min) {
                            min = sum;
                        }
                    }
                    new_vec[xj] = min;
                }
                //to judge if the vector has been converged
                bool flag = (M(node, neighbor_index) == new_vec);
                if (flag) {
                    isConverge(node, neighbor_index) = true;
                }
                else {
                    M(node, neighbor_index) = new_vec;
                }
            }

        }
    }
    cout << "finish iteration" << endl;

    //after iteration,we need to find the optimum label for every node
    for (int i = 0; i < unknown_size; i++) {
        cout << "i: " << i << endl;
        for (int k = 0; k < sample_size; k++) {
            sum_vec[k] = 0;
            sum_vec[k] += E1[i][k];
        }
//        initArray(sum_vec, sample_size);
//        addArray(sum_vec, E1[i], sum_vec,sample_size);
        for (int j = 0; j < unknown[i].neighbors.size(); j++) {
            //neighbor to node
            sum_vec += M(j, i);
//            for (int k = 0; k < sample_size; k++) {
//                sum_vec[k] += M[j][i][k];
//            }
//            addArray(sum_vec, M[j][i], sum_vec, sample_size);
        }
        //find the min label
        double min = FLT_MAX;
        int label_index = 0;
        for (int i = 0; i < sample_size; i++) {
            if (sum_vec[i] < min) {
                min = sum_vec[i];
                label_index = i;
            }
        }
        sample_label.push_back(label_index);
    }

    return sample_label;

}

void StructurePropagation::mergeCurves(vector<bool>&isSingle) {

    int num_curves = pointlist.size();

    //begin to merge
    for (int i = 0; i < num_curves - 1; i++) {
        if (unknown_anchors[i].size() == 0) {
            continue;
        }
        for (int j = i + 1; j < num_curves; j++) {
            if (unknown_anchors[j].size() == 0) {
                continue;
            }
            if (isIntersect(i, j)) {
                isSingle[i] = false;
                isSingle[j] = false;
                int unknown_size = MIN(unknown_anchors[i].size(), unknown_anchors[j].size());
                max_unknownsize = MAX(unknown_size, max_unknownsize);
                //transfer the unknown anchor points
                int num_points = pointlist[i].size();
                int num_unknown_size = unknown_anchors[i].size();
                for (int anchor_index = 0; anchor_index < unknown_anchors[j].size(); anchor_index++) {
                    unknown_anchors[j][anchor_index].anchor_point += num_points;
                    unknown_anchors[j][anchor_index].begin_point += num_points;
                    unknown_anchors[j][anchor_index].end_point += num_points;
                    //add the anchor point to the end of the first curve
                    for (int t = 0; t < unknown_anchors[j][anchor_index].neighbors.size(); t++) {
                        unknown_anchors[j][anchor_index].neighbors[t] += num_unknown_size;
                        unknown_anchors[j][anchor_index].curve_index = i;
                    }
                    unknown_anchors[i].push_back(unknown_anchors[j][anchor_index]);
                }
                //transfer the sample anchor points
                for (int sample_index = 0; sample_index < sample_anchors[j].size(); sample_index++) {
                    sample_anchors[j][sample_index].anchor_point += num_points;
                    sample_anchors[j][sample_index].begin_point += num_points;
                    sample_anchors[j][sample_index].end_point += num_points;
                    sample_anchors[j][sample_index].curve_index = i;
                    sample_anchors[i].push_back(sample_anchors[j][sample_index]);
                }
                //transfer the real points
                for (int point_index = 0; point_index < pointlist[j].size(); point_index++) {
                    pointlist[i].push_back(pointlist[j][point_index]);
                }

                unknown_anchors[j].clear();
                sample_anchors[j].clear();
                pointlist[j].clear();
            }
        }
    }

}

void StructurePropagation::getOneNewCurve(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample, int curve_index, bool flag, Mat &result) {
    vector<int>label;
    if (sample.size() == 0) {
        return;
    }
    if (flag) {
        label = DP(unknown, sample);
    }
    else {
        label = BP(unknown, sample, curve_index);
    }
    if (unknown.size() != label.size()) {
        cout << endl << "In getOneNewCurve() : The sizes of unknown and label are different" << endl;
        throw exception();
    }
    for (int i = 0; i < unknown.size(); i++) {
        Mat patch = getOnePatch(sample[label[i]], img, curve_index);
        copyPatchToImg(unknown[i], patch, result, curve_index);
    }
}



//for debug
void StructurePropagation::drawAnchors() {
    Mat showAnchors = img.clone();
    for (int i = 0; i < sample_anchors.size(); i++) {
        for (int j = 0; j < sample_anchors[i].size(); j++) {
            Point p = pointlist[i][sample_anchors[i][j].anchor_point];
            //circle(showAnchors, p, 5, Scalar(255, 255, 0));
            Point tmp = getLeftTopPoint(p);
            Rect rec(tmp.x,tmp.y, block_size, block_size);
            rectangle(showAnchors, rec, Scalar(255, 0, 0));
        }
    }
    for (int i = 0; i < unknown_anchors.size(); i++) {
        for (int j = 0; j < unknown_anchors[i].size(); j++) {
            Point p = pointlist[i][unknown_anchors[i][j].anchor_point];
            //circle(showAnchors, p, 5, Scalar(255, 255, 0));
            Point tmp = getLeftTopPoint(p);
            Rect rec(tmp.x, tmp.y, block_size, block_size);
            rectangle(showAnchors, rec, Scalar(255, 255, 0));
        }
    }
    imshow("show anchors", showAnchors);
    waitKey(0);
}

void StructurePropagation::Run(const Mat &mask, const Mat& img, Mat &mask_structure, vector<vector<Point>> &plist, Mat& result) {
    this->mask = mask;
    this->img = img;
    this->pointlist.clear();
    this->pointlist = plist;
    this->sample_anchors.clear();
    this->unknown_anchors.clear();
    this->isSingle.resize(plist.size());
    this->max_unknownsize = 0;
    for (int i = 0; i < isSingle.size(); i++){
        isSingle[i] = true;
    }
    getAnchors();
    drawAnchors();
    int curve_size = plist.size();
//    vector<bool>isSingle(curve_size, true);

    mergeCurves(isSingle);

    for (int i = 0; i < curve_size; i++) {
        if (isSingle[i]) {
            if (!unknown_anchors[i].empty())
                getOneNewCurve(unknown_anchors[i], sample_anchors[i], i, true, result);//DP
        }
        else {
            if(!unknown_anchors[i].empty())
                getOneNewCurve(unknown_anchors[i], sample_anchors[i], i, false, result);//BP
        }
    }


    // update mask (mark anchored patches as known)
    for (int i = 0; i < unknown_anchors.size(); i++)
    {
        for (auto p : unknown_anchors[i]){
            Point tar = pointlist[i][p.anchor_point];
            for (int j = -block_size / 2; j < block_size / 2; j++)
            {
                for (int k = -block_size / 2; k < block_size / 2; k++)
                {
                    int y = j + tar.y;
                    int x = k + tar.x;
                    if (x >= 0 && y >= 0 && x < mask_structure.cols && y < mask_structure.rows)
                    {
                        mask_structure.at<uchar>(y, x) = 255;
                    }
                }
            }
        }
    }

}


//get all the anchor points in the image and save them
void StructurePropagation::getAnchors() {
    vector<AnchorPoint>unknown, sample;
    for (int i = 0; i < pointlist.size(); i++) {
        getOneCurveAnchors(i, unknown, sample);
        this->unknown_anchors.push_back(unknown);
        this->sample_anchors.push_back(sample);

        sample.clear();
        unknown.clear();
    }
    cout << endl;
}

bool StructurePropagation::isBorder(Point p, bool isSample) {
//    Point p = pointlist[curve_index][anchor_index];
    int left = MAX(p.x - block_size / 2, 0);
    int right = MIN(p.x + block_size / 2, mask.cols);
    int up = MAX(p.y - block_size / 2, 0);
    int down = MIN(p.y + block_size / 2, mask.rows);
    for (int i = left; i < right; i++) {
        if (!mask.at<uchar>(up, i) == isSample || !mask.at<uchar>(down - 1, i) == isSample) {
            return true;
        }
    }

    for (int i = up; i < down; i++) {
        if (!mask.at<uchar>(i, left) == isSample || !mask.at<uchar>(i, right - 1) == isSample) {
            return true;
        }
    }
    return false;
}


void StructurePropagation::getOneCurveAnchors(int curve_index, vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample) {
    int num_points = pointlist[curve_index].size();
    PointType type;
    int cur_idx, next_idx;
    cur_idx = 0;
    bool border;
    int cur_unknown = 0;
    for (cur_idx = 0; cur_idx < num_points; cur_idx += sample_step) {
        // outside the image
        Point p = pointlist[curve_index][cur_idx];
        if (p.x - block_size / 2 < 0 || p.x + block_size / 2 >= img.cols || p.y - block_size / 2 < 0 || p.y + block_size / 2 >= img.rows){
            continue;
        }
        if (mask.at<uchar>(p) == 0) {
            border = isBorder(p, false);
            type = border ? BORDER : INNER;
        }
        else {
            border = isBorder(p, true);
            type = border ? BORDER : OUTER;
        }
        AnchorPoint anchor (cur_idx, block_size, type, curve_index);
        // unknown type
        if (type == BORDER || type == INNER) {
            if (cur_unknown - 1 >= 0) {
                anchor.neighbors.push_back(cur_unknown - 1);
                unknown[cur_unknown - 1].neighbors.push_back(cur_unknown);
            }
            unknown.push_back(anchor);
            cur_unknown += 1;
        }
        // sample type
        else {
            sample.push_back(anchor);
        }
    }
}

Mat StructurePropagation::getOnePatch(Point p, Mat &img) {
    Mat patch;
    Point left_top = getLeftTopPoint(p);
    Point right_buttom = left_top + Point(block_size, block_size);
    Rect rec(left_top, right_buttom);

    if (left_top.x<0 || left_top.y<0 || right_buttom.x>img.cols || right_buttom.y>img.rows) {
        cout << "exception:" << left_top << "   " << right_buttom << "when getting one patch" << endl;
        throw exception();
    }
    img(rec).copyTo(patch);
    return patch;
}

Mat StructurePropagation::getOnePatch(AnchorPoint ap, Mat &img, int curve_index) {
    Mat patch;
    Rect rec = getRect(ap, curve_index);
    if (rec.x<0 || rec.y<0 || rec.x + rec.width>img.cols || rec.y + rec.height>img.rows) {
        cout << "exception:" << rec.x << "   " << rec.y << "when getting one patch" << endl;
        throw exception();
    }
    img(rec).copyTo(patch);
    return patch;
}

void StructurePropagation::copyPatchToImg(AnchorPoint unknown, Mat &patch, Mat &img, int curve_index) {
    Rect rec = getRect(unknown, curve_index);
    //need to be correct ,to be done
    Mat correct_patch = patch.clone();
    Mat blend=pc->correct(correct_patch, mask, img,rec);
    blend.copyTo(img(rec));
}

Point StructurePropagation::getLeftTopPoint(Point p) {

    int x = (p.x - block_size / 2) > 0 ? p.x - block_size / 2 : 0;
    int y = (p.y - block_size / 2) > 0 ? p.y - block_size / 2 : 0;
    return Point(x, y);
}

Point StructurePropagation::getAnchorPoint(AnchorPoint ap, int curve_index) {
    return pointlist[curve_index][ap.anchor_point];
}

Rect StructurePropagation::getRect(AnchorPoint ap, int curve_index) {
    Point p = pointlist[curve_index][ap.anchor_point];
    Point left_top = p - Point(block_size/2, block_size/2);
    Point right_down = left_top + Point(block_size , block_size);
    return Rect(left_top, right_down);
}


void StructurePropagation::TextureCompletion(Mat _mask, Mat LineMask, const Mat &mat, Mat &result)
{
    int N = _mask.rows;
    int M = _mask.cols;
    int knowncount = 0;
    for (int i = 0; i < N;i++)
        for (int j = 0; j < M; j++)
        {
            knowncount += (_mask.at<uchar>(i, j) == 255);
        }
    if (knowncount * 2< N * M)
    {
        for (int i = 0; i < N;i++)
            for (int j = 0; j < M; j++)
                _mask.at<uchar>(i, j) = 255 - _mask.at<uchar>(i, j);
    }

    vector<vector<int> >my_mask(N, vector<int>(M, 0)), sum_diff(N, vector<int>(M, 0));

    for (int i = 0; i < N;i++)
        for (int j = 0; j < M; j++)
            LineMask.at<uchar>(i, j) = LineMask.at<uchar>(i, j) * 100;

    result = mat.clone();
//    imshow("mask", _mask);
//    imshow("linemask", LineMask);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
        {
            my_mask[i][j] = (_mask.at<uchar>(i, j) == 255);
            if (my_mask[i][j] == 0 && LineMask.at<uchar>(i, j) > 0)
            {
                my_mask[i][j] = 2;
            }
        }
    int bs = 5;
    int step = 1 * bs;
    auto usable(my_mask);
    int to_fill = 0, filled = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
        {
            to_fill += (my_mask[i][j] == 0);
        }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
        {
            if (my_mask[i][j] == 1)
                continue;
            int k0 = max(0, i - step), k1 = min(N - 1, i + step);
            int l0 = max(0, j - step), l1 = min(M - 1, j + step);
            for (int k = k0; k <= k1; k++)
                for (int l = l0; l <= l1; l++)
                    usable[k][l] = 2;
        }
    Mat use = _mask.clone();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            if (usable[i][j] == 2)
                use.at<uchar>(i, j) = 255;
            else use.at<uchar>(i, j) = 0;
//    imshow("usable", use);
    int itertime = 0;
    Mat match;
    while (true)
    {
        itertime++;
        int x, y, cnt = -1;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < M; j++)
            {
                if (my_mask[i][j] != 0) continue;
                bool edge = false;
                int k0 = max(0, i - 1), k1 = min(N - 1, i + 1);
                int l0 = max(0, j - 1), l1 = min(M - 1, j + 1);
                for (int k = k0; k <= k1;k++)
                    for (int l = l0; l <= l1; l++)
                        edge |= (my_mask[k][l] == 1);
                if (!edge) continue;
                k0 = max(0, i - bs), k1 = min(N - 1, i + bs);
                l0 = max(0, j - bs), l1 = min(M - 1, j + bs);
                int tmpcnt = 0;
                for (int k = k0; k <= k1; k++)
                    for (int l = l0; l <= l1; l++)
                        tmpcnt += (my_mask[k][l] == 1);
                if (tmpcnt > cnt)
                {
                    cnt = tmpcnt;
                    x = i;
                    y = j;
                }
            }
        if (cnt == -1) break;

        int k0 = min(x, bs), k1 = min(N - 1 - x, bs);
        int l0 = min(y, bs), l1 = min(M - 1 - y, bs);
        int sx, sy, min_diff = INT_MAX;
        for (int i = step; i + step < N; i += step) {
            for (int j = step; j + step < M; j += step) {
                if (usable[i][j] == 2)continue;
                int tmp_diff = 0;
                for (int k = -k0; k <= k1; k++)
                    for (int l = -l0; l <= l1; l++)
                    {
                        if (my_mask[x + k][y + l] != 0)
//                            tmp_diff += dist(result.at<Vec3b>(i + k, j + l), result.at<Vec3b>(x + k, y + l));
                            tmp_diff += norm(result.at<Vec3b>(i + k, j + l), result.at<Vec3b>(x + k, y + l));
                    }
                sum_diff[i][j] = tmp_diff;
                if (min_diff > tmp_diff)
                {
                    sx = i;
                    sy = j;
                    min_diff = tmp_diff;
                }
            }
        }

        printf("done :%.2lf%%\n", 100.0 * filled / to_fill);
        imshow("run", result);
        waitKey(10);
    }
}