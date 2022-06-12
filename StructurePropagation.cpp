#include"StructurePropagation.h"
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
    if (unknown.curve_index != sample.curve_index) {
        cout << "calEs error: not in the same curve" << endl;
        throw exception();
    }
    int curve_index = unknown.curve_index;
    Point p1 = pointlist[curve_index][unknown.anchor_point];
    Point p2 = pointlist[curve_index][sample.anchor_point];
    Point origin_unknown = getLeftTopPoint(p1);
    Point origin_sample = getLeftTopPoint(p2);

    vector<Point>ci, cxi;
    // calculate the relative coordinates
    for (int i = unknown.anchor_point - block_size / 2; i <= unknown.anchor_point + block_size / 2; i++) {
        Point p = pointlist[curve_index][i];
        ci.push_back(p - origin_unknown);
    }
    for (int i = sample.anchor_point - block_size / 2; i <= sample.anchor_point + block_size / 2; i++) {
        Point p = pointlist[curve_index][i];
        cxi.push_back(p - origin_sample);
    }

    // calculate distance and do normalization
    int num_ci = block_size + 1;
    int num_cxi = block_size + 1;
    double d1 = calDistance(ci, cxi) / num_ci;
    double d2 = calDistance(ci, cxi) / num_cxi;

    return d1 + d2;
}

double StructurePropagation::calEi(AnchorPoint unknown, AnchorPoint sample) {
    // Ei of the inner unknown AnchorPoints is set to 0
    if (unknown.type != BOUNDARY)
        return 0;

    if (unknown.curve_index != sample.curve_index) {
        cout << "calEi error: not in the same curve" << endl;
        throw exception();
    }
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
    if (unknown1.curve_index != sample1.curve_index) {
        cout << "calE2 error: not in the same curve" << endl;
        throw exception();
    }

    int curve_index = unknown1.curve_index;

    Point u1_point = pointlist[curve_index][unknown1.anchor_point];
    Point u1_lefttop = getLeftTopPoint(u1_point);
    Rect rec1(u1_lefttop.x, u1_lefttop.y, unknown1.block_size, unknown1.block_size);

    Point u2_point = pointlist[curve_index][unknown2.anchor_point];
    Point u2_lefttop = getLeftTopPoint(u2_point);
    Rect rec2(u2_lefttop.x, u2_lefttop.y, unknown2.block_size, unknown2.block_size);

    // get the overlap area
    Rect intersect = rec1 & rec2;

    Mat patch1 = getOnePatch(getPatch(sample1, curve_index), img);
    Mat patch2 = getOnePatch(getPatch(sample2, curve_index), img);

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
    cout << "start DP ..." << endl;
    int unknown_size = unknown.size();
    int sample_size = sample.size();

    if (unknown_size == 0 || sample_size == 0) {
        cout << "DP exception" << endl;
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
    for (int i = 0; i < unknown_size; i++) {
        cout << sample_label[i] << ' ';
    }
    cout << endl;
    cout << "finish DP" << endl;
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
                // update neighbors of the anchor point in curve1 (add j and j+1 into it)
                unknown_anchors[curve1][i].neighbors.push_back(j + num_curve1);
                unknown_anchors[curve1][i].neighbors.push_back(j + 1 + num_curve1);
                // update neighbors of the anchor point j and j + 1 in curve2
                // the neighbor of it should be updated to point1
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

vector<int> StructurePropagation::BP(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample) {
    cout << "Start BP ... ..." << endl;

    vector<int>sample_label;
    int unknown_size = unknown.size();
    int sample_size = sample.size();

    // M[unknown_size][unknown_size][sample_size]
    Matrix<VectorX<double>, Dynamic, Dynamic> M;
    M.resize(unknown_size, unknown_size);
    VectorX<double> M_in(sample_size);
    // initialize M[unknown_size][unknown_size] = 0
    M_in.fill(0);
    M.fill(M_in);

    // E1[unknown_size][sample_size]
    VectorX<VectorX<double>> E1(unknown_size);
    VectorX<double> E1_in(sample_size);
    // initialize E1 = 0
    E1_in.fill(0);
    E1.fill(E1_in);

    // calculate the E1 matrix
    for (int i = 0; i < unknown_size; i++) {
        for (int xi = 0; xi < sample_size; xi++) {
            E1[i][xi] = calE1(unknown[i], sample[xi]);
        }
    }

    // isConverge[unknown_size][unknown_size]: judge if it's converge
    Matrix<bool, Dynamic, Dynamic> isConverge;
    isConverge.resize(unknown_size, unknown_size);
    isConverge.fill(false);

    // sum of M_ki (neighbors around)
    VectorX<double> sum_M(sample_size);
    // E1 + sum_M
    VectorX<double> E_M_sum(sample_size);
    // M_t (update)
    VectorX<double> new_M(sample_size);

    //begin to iterate
    cout << "unknown_size:" << max_unknownsize << endl;
    // iteration number: T (the maximum distance between any two nodes in the graph)
    for (int t = 0; t < max_unknownsize; t++) {
        cout << "t: " << t << endl;
        for (int node = 0; node < unknown_size; node++) {
            //calculate the sum of M[t-1][i][j]
            sum_M.fill(0);
            for (int k = 0; k < unknown[node].neighbors.size(); k++) {
                // sum of all neighbor_index k
                sum_M += M(k, node);
            }
            // outer loop M(i,j)
            for (int j = 0; j < unknown[node].neighbors.size(); j++) {
                int neighbor_index = unknown[node].neighbors[j];
                // is converge
                if (isConverge(node, neighbor_index)) {
                    continue;
                }
                // the sum item should not include the current neighbor (k != j)
                sum_M -= M(neighbor_index, node);
                E_M_sum = E1[node] + sum_M;

                // find the sample label to make (E1 + E2 + sum_M) to be minimum
                for (int xj = 0; xj < sample_size; xj++) {
                    double min = DBL_MAX;
                    for (int xi = 0; xi < sample_size; xi++) {
                        double E2 = calE2(unknown[node], unknown[neighbor_index], sample[xi], sample[xj]);
                        double sum = E2 + E_M_sum[xi];
                        if (sum < min) {
                            min = sum;
                        }
                    }
                    new_M[xj] = min;
                }
                //to judge if the vector has been converged
                bool flag = (M(node, neighbor_index) == new_M);
                if (flag) {
                    isConverge(node, neighbor_index) = true;
                }
                else {
                    M(node, neighbor_index) = new_M;
                }
            }

        }
    }
    cout << "finish iteration" << endl;

    // find the optimal label after iteration
    // xi = argmin{E1 + sumM}
    for (int i = 0; i < unknown_size; i++) {
        cout << "i: " << i << endl;
        sum_M = E1[i];
        for (int k = 0; k < unknown[i].neighbors.size(); k++) {
            // add all the neighbors
            sum_M += M(k, i);
        }
        //find the min label
        double min = FLT_MAX;
        int label_index = 0;
        for (int i = 0; i < sample_size; i++) {
            if (sum_M[i] < min) {
                min = sum_M[i];
                label_index = i;
            }
        }
        sample_label.push_back(label_index);
    }

    for (int i = 0; i < sample_label.size(); i++) {
        cout << sample_label[i] << " ";
    }

    cout << "finish BP" << endl;
    return sample_label;
}

void StructurePropagation::mergeCurves() {
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
            // merge intersected lines
            if (isIntersect(i, j)) {
                isSingle[i] = false;
                isSingle[j] = false;
                // update the pointlist index
                for (int point_index = 0; point_index < pointlist[j].size(); point_index++) {
                    pointlist[i].push_back(pointlist[j][point_index]);
                }
                pointlist[j].clear();
                // set the maximum distance of any two nodes in the unknown area
                int unknown_size = MIN(unknown_anchors[i].size(), unknown_anchors[j].size());
                max_unknownsize = MAX(unknown_size, max_unknownsize);
                // update the unknown anchor index (include the anchor_point, neighbor index and curve_index)
                int num_points = pointlist[i].size();
                int num_unknown_size = unknown_anchors[i].size();
                for (int unknown_index = 0; unknown_index < unknown_anchors[j].size(); unknown_index++) {
                    unknown_anchors[j][unknown_index].anchor_point += num_points;
                    unknown_anchors[j][unknown_index].curve_index = i;
                    for (auto & n: unknown_anchors[j][unknown_index].neighbors) {
                        n += num_unknown_size;
                    }
                    unknown_anchors[i].push_back(unknown_anchors[j][unknown_index]);
                }
                unknown_anchors[j].clear();
                // update the sample anchor index (include the anchor_point, neighbor index and curve_index)
                for (int sample_index = 0; sample_index < sample_anchors[j].size(); sample_index++) {
                    sample_anchors[j][sample_index].anchor_point += num_points;
                    sample_anchors[j][sample_index].curve_index = i;
                    sample_anchors[i].push_back(sample_anchors[j][sample_index]);
                }
                sample_anchors[j].clear();
            }
        }
    }

}

void StructurePropagation::drawAnchors() {
    Mat showAnchors = img.clone();
    // draw samples
    for (int i = 0; i < sample_anchors.size(); i++) {
        for (int j = 0; j < sample_anchors[i].size(); j++) {
            Point p = pointlist[i][sample_anchors[i][j].anchor_point];
            Point tmp = getLeftTopPoint(p);
            Rect rec(tmp.x,tmp.y, block_size, block_size);
            rectangle(showAnchors, rec, Scalar(255, 0, 0));
        }
    }
    // draw unknowns
    for (int i = 0; i < unknown_anchors.size(); i++) {
        for (int j = 0; j < unknown_anchors[i].size(); j++) {
            Point p = pointlist[i][unknown_anchors[i][j].anchor_point];
            Point tmp = getLeftTopPoint(p);
            Rect rec(tmp.x, tmp.y, block_size, block_size);
            rectangle(showAnchors, rec, Scalar(255, 255, 0));
        }
    }
    imshow("show anchors", showAnchors);
    waitKey(0);
    destroyWindow("show anchors");
}


void StructurePropagation::Run(const Mat &mask, const Mat& img, Mat &mask_structure, vector<vector<Point>> &plist, Mat& result) {
    // initialize
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
    // get anchors in the image
    getAnchors();
    drawAnchors();
    int curve_size = plist.size();
    mergeCurves();

    // begin to run
    for (int i = 0; i < curve_size; i++) {
        if (unknown_anchors[i].empty() || sample_anchors[i].empty()){
            continue;
        }
        vector<int> label;
        if (isSingle[i]) {
            label = DP(unknown_anchors[i], sample_anchors[i]);
        }
        else {
            label = BP(unknown_anchors[i], sample_anchors[i]);
        }
        // copy sample to unknown
        for (int j = 0; j < unknown_anchors[i].size(); j++) {
            Mat patch = getOnePatch(sample_anchors[i][label[j]], img, i);
            copyPatchToImg(unknown_anchors[i][j], patch, result, i);
        }
    }

    // update mask (mark the unknown mask to known)
    for (int i = 0; i < unknown_anchors.size(); i++) {
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

// get patches in the image
void StructurePropagation::getAnchors() {
    vector<AnchorPoint>unknown, sample;
    for (int i = 0; i < pointlist.size(); i++) {
        getAnchorOneCurve(unknown, sample, i);
        this->unknown_anchors.push_back(unknown);
        this->sample_anchors.push_back(sample);
        sample.clear();
        unknown.clear();
    }
}

// judge if the point is near the boundary
bool StructurePropagation::isBoundary(Point p, bool isSample) {
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


void StructurePropagation::getAnchorOneCurve(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample, int curve_index) {
    int num_points = pointlist[curve_index].size();
    AnchorType type;
    int cur_idx;
    bool boundary;
    int cur_unknown = 0;
    for (cur_idx = 0; cur_idx < num_points; cur_idx += sample_step) {
        // outside the image
        Point p = pointlist[curve_index][cur_idx];
        if (p.x - block_size / 2 < 0 || p.x + block_size / 2 >= img.cols || p.y - block_size / 2 < 0 || p.y + block_size / 2 >= img.rows){
            continue;
        }
        if (mask.at<uchar>(p) == 0) {
            boundary = isBoundary(p, false);
            type = boundary ? BOUNDARY : INNER;
        }
        else {
            boundary = isBoundary(p, true);
            type = boundary ? BOUNDARY : OUTER;
        }
        AnchorPoint anchor (cur_idx, block_size, type, curve_index);
        // unknown type
        if (type == BOUNDARY || type == INNER) {
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

Mat StructurePropagation::getOnePatch(Point p, Mat img) {
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

Mat StructurePropagation::getOnePatch(AnchorPoint ap, Mat img, int curve_index) {
    Mat patch;
    Rect rec = getRect(ap, curve_index);
    if (rec.x<0 || rec.y<0 || rec.x + rec.width > img.cols || rec.y + rec.height > img.rows) {
        cout << "getOnePatch exception:" << rec.x << " " << rec.y << endl;
        throw exception();
    }
    img(rec).copyTo(patch);
    return patch;
}



Mat StructurePropagation::PhotometricCorrection(Mat &patch, Mat &mask, Mat &img, Rect &rec) {
    Mat dst = img(rec).clone();
    Mat src = patch.clone();
    Mat _mask = mask(rec).clone();

    threshold(mask(rec), _mask, 100, 255, CV_THRESH_BINARY_INV);

    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            if (_mask.at<uchar>(i, j) == 255) {
                dst.at<Vec3b>(i, j) = patch.at<Vec3b>(i, j);
            }
        }
    }
    Rect re;
    re.x = 1; re.y = 1;
    re.width = rec.width - 1;
    re.height = rec.height - 1;
    _mask.setTo(255);

    Mat blend;
    src = src(re).clone();
    seamlessClone(src, dst, _mask, Point(patch.cols / 2, patch.rows / 2), blend, NORMAL_CLONE);
    blend.copyTo(patch);
    mask(rec).setTo(255);
    return blend;
}

void StructurePropagation::copyPatchToImg(AnchorPoint unknown, Mat &patch, Mat &img, int curve_index) {
    Rect rec = getRect(unknown, curve_index);
    Mat correct_patch = patch.clone();
    Mat blend = PhotometricCorrection(correct_patch, mask, img, rec);
    blend.copyTo(img(rec));
}

Point StructurePropagation::getLeftTopPoint(Point p) {

    int x = (p.x - block_size / 2) > 0 ? p.x - block_size / 2 : 0;
    int y = (p.y - block_size / 2) > 0 ? p.y - block_size / 2 : 0;
    return Point(x, y);
}

Point StructurePropagation::getPatch(AnchorPoint ap, int curve_index) {
    return pointlist[curve_index][ap.anchor_point];
}

Rect StructurePropagation::getRect(AnchorPoint ap, int curve_index) {
    Point p = pointlist[curve_index][ap.anchor_point];
    Point left_top = p - Point(block_size/2, block_size/2);
    Point right_down = left_top + Point(block_size , block_size);
    return Rect(left_top, right_down);
}


void StructurePropagation::TextureCompletion(Mat _mask, Mat structureLine, const Mat &mat, Mat &result) {
    cout << "TextureCompletion..." << endl;
    int N = _mask.rows;
    int M = _mask.cols;

    threshold(_mask, _mask, 100, 255, THRESH_BINARY_INV);
    vector<vector<int> >my_mask(N, vector<int>(M, 0));

    result = mat.clone();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            my_mask[i][j] = (_mask.at<uchar>(i, j) == 255);
            if (my_mask[i][j] == 0 && structureLine.at<uchar>(i, j) > 0) {
                my_mask[i][j] = 2;
            }
        }
    }

    int step = 10;
    auto record(my_mask);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (my_mask[i][j] == 1) {
                continue;
            }
            int k0 = max(0, i - step), k1 = min(N - 1, i + step);
            int l0 = max(0, j - step), l1 = min(M - 1, j + step);
            // record the structure line
            for (int k = k0; k <= k1; k++) {
                for (int l = l0; l <= l1; l++) {
                    record[k][l] = 2;
                }
            }
        }
    }

    while (true) {
        int x, y, cnt = -1;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                if (my_mask[i][j] != 0) continue;
                // find the edge
                bool edge = false;
                int k0 = max(0, i - 1), k1 = min(N - 1, i + 1);
                int l0 = max(0, j - 1), l1 = min(M - 1, j + 1);
                for (int k = k0; k <= k1;k++) {
                    for (int l = l0; l <= l1; l++) {
                        edge |= (my_mask[k][l] == 1);
                    }
                }

                if (!edge) continue;
                k0 = max(0, i - step), k1 = min(N - 1, i + step);
                l0 = max(0, j - step), l1 = min(M - 1, j + step);
                int tmpcnt = 0;
                for (int k = k0; k <= k1; k++) {
                    for (int l = l0; l <= l1; l++) {
                        tmpcnt += (my_mask[k][l] == 1);
                    }
                }
                if (tmpcnt > cnt) {
                    cnt = tmpcnt;
                    x = i;
                    y = j;
                }
            }
        }

        // finish filling
        if (cnt == -1) break;

        int k0 = min(x, step), k1 = min(N - 1 - x, step);
        int l0 = min(y, step), l1 = min(M - 1 - y, step);
        int sx, sy, min_diff = INT_MAX;
        // find the optimal patch
        for (int i = step; i + step < N; i += step)
            for (int j = step; j + step < M; j += step) {
                // skip the structure line
                if (record[i][j] == 2) continue;
                int tmp_diff = 0;
                // find the patch with minimum difference
                for (int k = -k0; k <= k1; k++) {
                    for (int l = -l0; l <= l1; l++) {
                        if (my_mask[x + k][y + l] != 0)
                            tmp_diff += norm(result.at<Vec3b>(i + k, j + l), result.at<Vec3b>(x + k, y + l));
                    }
                }
                // update
                if (min_diff > tmp_diff) {
                    sx = i;
                    sy = j;
                    min_diff = tmp_diff;
                }
            }

        // fill the mask
        for (int k = -k0; k <= k1; k++) {
            for (int l = -l0; l <= l1; l++) {
                if (my_mask[x + k][y + l] == 0) {
                    result.at<Vec3b>(x + k, y + l) = result.at<Vec3b>(sx + k, sy + l);
                    my_mask[x + k][y + l] = 1;
                }
            }
        }

        imshow("run", result);
        waitKey(10);
    }
    cout << "finish TextureCompletion" << endl;
}