#include"StructurePropagation.h"
#include"math_function.h"

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
    matchTemplate(m1, m2, result, CV_TM_SQDIFF_NORMED);
    return result.at<double>(0, 0);
}

double StructurePropagation::calDistance(vector<Point>ci, vector<Point>cxi) {
    double result = 0;
    double shortest, sq;

    double normalized = norm(Point(block_size, block_size));

    for (int i = 0; i < ci.size(); i++) {
        shortest = FLT_MAX;
        for (int j = 0; j < cxi.size(); j++) {
            sq = norm(ci[i] - cxi[j]) / normalized;
            sq *= sq;
            if (sq < shortest) {
                shortest = sq;
            }
        }
        result += shortest;
    }
    return result;
}

double StructurePropagation::calEs(AnchorPoint unknown, AnchorPoint sample, int curve_index) {
    vector<Point>ci, cxi;
    //the left_top point of the patches
    Point origin_unknown = getLeftTopPoint(unknown.anchor_point, curve_index);
    Point origin_sample = getLeftTopPoint(sample.anchor_point, curve_index);

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

double StructurePropagation::calEi(AnchorPoint unknown, AnchorPoint sample, int curve_index) {
    if (unknown.type != BORDER)
        return 0;
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

double StructurePropagation::calE1(AnchorPoint unknown, AnchorPoint sample, int curve_index) {
    //return KS * calEs(unknown, sample, curve_index) + KI * calEi(unknown, sample, curve_index);
    double  Es = calEs(unknown, sample, curve_index);
    double  Ei = calEi(unknown, sample, curve_index);
    //cout << "Es=" << KS*Es << "  " << "Ei=" << KI*Ei << endl;
    return ks * Es + ki * Ei;
}


double StructurePropagation::calE2(AnchorPoint unknown1, AnchorPoint unknown2, AnchorPoint sample1, AnchorPoint sample2, int curve_index) {
    //get the four vertexes of the two patches
    Point ult = getLeftTopPoint(unknown1.anchor_point, curve_index);
    Point urb = ult + Point(block_size, block_size);

    Point slt = getLeftTopPoint(unknown2.anchor_point, curve_index);
    Point srb = slt + Point(block_size, block_size);

    Rect rec1(ult, srb);
    Rect rec2(urb, slt);
    Rect intersect = rec1 & rec2;

    Mat patch1 = getOnePatch(getAnchorPoint(sample1, curve_index), img);
    Mat patch2 = getOnePatch(getAnchorPoint(sample2, curve_index), img);

    Mat copy1 = img.clone();
    Mat copy2 = img.clone();
    //overlap the srcimage with the corresponding sample patch
    patch1.copyTo(copy1(Rect(ult, urb)));
    patch2.copyTo(copy2(Rect(slt, srb)));

    double result;
    //callate the SSD of the overlap parts of two sample patches
    result = calSSD(copy1(intersect), copy2(intersect));
    //for debug
    //cout << "E2=" << result << endl;
    return result;
}

vector<int> StructurePropagation::DP(vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample, int curve_index) {

    int unknown_size = unknown.size();
    int sample_size = sample.size();

    if (unknown_size == 0 || sample_size == 0) {
        cout << "In DP: the size of vector AnchorPoint is 0" << endl;
        throw exception();
    }

    double **M = new double*[unknown_size];
    int **last_point = new int*[unknown_size];
    double **E1 = new double*[unknown_size];//unknonw_size*sample_size

    for (int i = 0; i < unknown_size; i++) {
        M[i] = new double[sample_size];
        last_point[i] = new int[sample_size];
        E1[i] = new double[sample_size];
    }


    for (int i = 0; i < unknown_size; i++) {
        for (int j = 0; j < sample_size; j++) {
            E1[i][j] = calE1(unknown[i], sample[j], curve_index);
        }
    }
    //initialize M[0]
    for (int i = 0; i < sample_size; i++) {
        M[0][i] = E1[0][i];
    }
    //callate the M[i][j]
    for (int i = 1; i < unknown_size; i++) {
        for (int j = 0; j < sample_size; j++) {
            double min = FLT_MAX;
            int min_index = 0;
            double E_1 = E1[i][j];
            // find the sample anchor t to make the Mi to be mininum
            for (int t = 0; t < sample_size; t++) {
                double tmp = calE2(unknown[i - 1], unknown[i], sample[t], sample[j], curve_index) + M[i - 1][t];
                if (tmp < min) {
                    min = tmp;
                    min_index = t;
                }
            }
            M[i][j] = E_1 + min;
            last_point[i][j] = min_index;
        }
    }
    vector<int>label;
    // find the best patch for the last unknown anchor point
    int last_patch = 0;
    double tmp_min = M[unknown_size - 1][0];
    for (int i = 0; i < sample_size; i++) {
        if (M[unknown_size - 1][i] < tmp_min) {
            last_patch = i;
            tmp_min = M[unknown_size - 1][i];
        }
    }
    label.push_back(last_patch);
    //back tracing
    if (unknown_size > 1) {
        for (int i = unknown_size - 1; i > 0; i--) {
            last_patch = last_point[i][last_patch];
            label.push_back(last_patch);
        }
    }

    reverse(label.begin(), label.end());
    for (int i = 0; i < unknown_size; i++) {
        delete[] M[i];
        delete[] last_point[i];
        delete[] E1[i];
    }
    delete[] M;
    delete[] E1;
    delete[] last_point;
    //for debug
    if (true) {
        cout << "The min energy of curve " << curve_index << " is " << tmp_min << endl;
        cout << "The size of the sample patch: " << label.size() << endl;
        for (int i = 0; i < label.size(); i++) {
            cout << label[i] << " ";
        }
        cout << endl;
    }

    cout << "DP is done" << endl;

    return label;
}


bool StructurePropagation::isNeighbor(Point point1, Point point2) {
    return norm(point1 - point2) < norm(Point(block_size / 2, block_size / 2));
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

    vector<int>label;
    int unknown_size = unknown.size();
    int sample_size = sample.size();

    double ***M = new double**[unknown_size];//unknown_times*unknown_size*sample_size
    double **E1 = new double*[unknown_size];//unknonw_size*sample_size

    //initilize the array
    for (int i = 0; i < unknown_size; i++) {
        E1[i] = new double[sample_size];
        M[i] = new double*[unknown_size];
        for (int j = 0; j < unknown_size; j++) {
            M[i][j] = new double[sample_size];
            initArray(M[i][j], sample_size);
        }
    }

    //to callate the matrix E1 for the convenience of the next callation
    for (int i = 0; i < unknown_size; i++) {
        for (int j = 0; j < sample_size; j++) {
            E1[i][j] = calE1(unknown[i], sample[j], curve_index);
        }
    }


    //to judge if the node has been converged
    bool **isConverge = new bool*[unknown_size];
    for (int i = 0; i < unknown_size; i++) {
        isConverge[i] = new bool[unknown_size];
        initArray(isConverge[i], unknown_size);
    }


    double *sum_vec = new double[sample_size];//the sum of vectors from neighbors
    double *E_M_sum = new double[sample_size];//sum_vec-M[j][i]+E[i]
    double *new_vec = new double[sample_size];//the final vector callated for M[i][j]

    //begin to iterate
    cout << "unknown_size:" << unknown_size << endl;
    for (int t = 0; t < unknown_size; t++) {
        cout << "t: " << t << endl;
        for (int node = 0; node < unknown_size; node++) {
            //calcaulate the sum of M[t-1][i][j]
            initArray(sum_vec, sample_size);
            for (int neighbor_index = 0; neighbor_index < unknown[node].neighbors.size(); neighbor_index++) {
                //neighbors to node
                addArray(sum_vec, M[neighbor_index][node],sum_vec,sample_size);
            }
            //node to neighbors
            for (int times = 0; times < unknown[node].neighbors.size(); times++) {
                int neighbor_index = unknown[node].neighbors[times];
                if (isConverge[node][neighbor_index] == true) {
                    continue;
                }
                minusArray(sum_vec, M[neighbor_index][node], E_M_sum, sample_size);
                addArray(E_M_sum, E1[node], E_M_sum,sample_size);

//                cout << "sample_size = " << sample_size << endl;
                for (int xj = 0; xj < sample_size; xj++) {
//                    cout << "xj: " << xj << endl;
                    double min = FLT_MAX;
                    for (int xi = 0; xi < sample_size; xi++) {
                        double E2 = calE2(unknown[node], unknown[neighbor_index], sample[xi], sample[xj], curve_index);
                        double sum = E2 + E_M_sum[xi];
                        if (sum < min) {
                            min = sum;
                        }
                    }
                    new_vec[xj] = min;
                }
                //to judge if the vector has been converged
                bool flag = isEqualArray(M[node][neighbor_index], new_vec,sample_size);
                if (flag) {
                    isConverge[node][neighbor_index] = true;
                }
                else {
                    moveArray(M[node][neighbor_index], new_vec, sample_size);
                }
            }

        }
    }
    cout << "finish iteration" << endl;

    //after iteration,we need to find the optimum label for every node
    for (int i = 0; i < unknown_size; i++) {
        cout << "i: " << i << endl;
        initArray(sum_vec, sample_size);
        addArray(sum_vec, E1[i], sum_vec,sample_size);
        for (int j = 0; j < unknown[i].neighbors.size(); j++) {
            //neighbor to node
            addArray(sum_vec, M[j][i], sum_vec, sample_size);
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
        label.push_back(label_index);
    }
    delete[] new_vec;
    delete[] sum_vec;
    delete[] E_M_sum;
    for (int i = 0; i < unknown_size; i++) {
        delete[] isConverge[i];
        delete[] E1[i];
        for (int j = 0; j < unknown_size; j++) {
            delete[]M[i][j];
        }
    }
    delete[] isConverge;
    delete[] E1;
    for (int i = 0; i < unknown_size; i++) {
        delete[]M[i];
    }
    delete[] M;
    return label;

}

void StructurePropagation::addNeighborFB(int curve_index) {
    for (int i = 0; i < unknown_anchors[curve_index].size(); i++) {
        if (i - 1 >= 0) {
            unknown_anchors[curve_index][i].neighbors.push_back(i - 1);
        }
        if (i + 1 < unknown_anchors[curve_index].size()) {
            unknown_anchors[curve_index][i].neighbors.push_back(i + 1);
        }
    }
}

void StructurePropagation::mergeCurves(vector<bool>&isSingle) {

    int num_curves = pointlist.size();
    //initialize the neighbors
    for (int i = 0; i < num_curves; i++) {
        addNeighborFB(i);
    }

    //begin to merge
    for (int i = 0; i < num_curves; i++) {
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
                    }
                    unknown_anchors[i].push_back(unknown_anchors[j][anchor_index]);
                }
                //transfer the sample anchor points
                for (int sample_index = 0; sample_index < sample_anchors[j].size(); sample_index++) {
                    sample_anchors[j][sample_index].anchor_point += num_points;
                    sample_anchors[j][sample_index].begin_point += num_points;
                    sample_anchors[j][sample_index].end_point += num_points;

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
        label = DP(unknown, sample, curve_index);
    }
    else {
        label = BP(unknown, sample, curve_index);
    }
    if (unknown.size() != label.size()) {
        cout << endl << "In getOneNewCurve() : The sizes of unknown and label are different" << endl;
        throw exception();
    }
    for (int i = 0; i < unknown.size(); i++) {
        cout << "patch: " << i << endl;
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
            rectangle(showAnchors, rec, Scalar(255, 255, 0));
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
    this->pointlist = plist;
    getAnchors();
    drawAnchors();
    int curve_size = plist.size();
    vector<bool>isSingle(curve_size, true);

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

//get all the anchor points on the one curve
int StructurePropagation::getOneAnchorFront(int lastanchor_index, PointType &t, int curve_index, bool flag, vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample) {
    Point p = pointlist[curve_index][lastanchor_index];
    Rect rec = getRect(p);
    int i = lastanchor_index + 1;
    if (i >= pointlist[curve_index].size() - 1) {
        return pointlist[curve_index].size() - 1;
    }
    if (mask.at<uchar>(pointlist[curve_index][i]) == 0) {
        t = INNER;
    }
    else {
        t = OUTER;
    }
    while (i < pointlist[curve_index].size() && contain(rec,pointlist[curve_index][i])) {
        uchar tmp = mask.at<uchar>(pointlist[curve_index][i]);
        if (tmp == 0 && t == OUTER || tmp == 255 && t == INNER) {
            t = BORDER;
            if (flag) {
                int count = sample.size();
                if (count>0)
                    sample[count - 1].type = BORDER;
            }
            else {
                int count = unknown.size();
                if (count>0)
                    unknown[count - 1].type = BORDER;
            }
        }
        i++;
    }
    return i;
}
//int StructurePropagation::getOneAnchorBack(int lastanchor_index, PointType &t, int curve_index, bool flag, vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample) {
//    Point p = pointlist[curve_index][lastanchor_index];
//    Rect rec = getRect(p);
//    int i = lastanchor_index - 1;
//    t = OUTER;
//    while (i >= 0 && contain(rec, pointlist[curve_index][i])) {
//        i--;
//    }
//    if (i < 0) {
//        return -1;
//    }
//    return i;
//}
void StructurePropagation::getOneCurveAnchors(int curve_index, vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample){
    //Point unknown_begin;
    int unknown_begin;
    int num_points = pointlist[curve_index].size();
    for (int i = 0; i < num_points; i++) {
        if (mask.at<uchar>(pointlist[curve_index][i]) == 0) {
            unknown_begin = i;
            break;
        }
    }
    PointType type;
    bool flag = true;
    //find all the unknown anchor points
    int now_index, last_index;

    last_index = unknown_begin;
    while (true) {
        if (last_index < 0) {
            break;
        }
//        now_index = getOneAnchorBack(last_index, type, curve_index, flag, unknown, sample);
        now_index = last_index - sample_step;
        if (now_index < 0) {
            break;
        }
        if (last_index != unknown_begin) {
            sample[sample.size() - 1].begin_point = now_index + 1;
        }
        AnchorPoint anchor(now_index, last_index-1, now_index, type);
        sample.push_back(anchor);
        last_index = now_index;
    }
    //this point doesn't have enough points in the patch
    if(sample.size()>0){
        sample.pop_back();
        reverse(sample.begin(), sample.end());
    }


//    int first_unknown_begin = getOneAnchorBack(unknown_begin, type, curve_index, flag, unknown, sample) + 1;
    int first_unknown_begin = unknown_begin - sample_step + 1;
    now_index = unknown_begin;
    last_index = unknown_begin;
    AnchorPoint anchor(first_unknown_begin, now_index, now_index, BORDER);
    unknown.push_back(anchor);

    while (true) {
        now_index = getOneAnchorFront(now_index, type, curve_index, flag, unknown, sample);
        if (now_index >= pointlist[curve_index].size() - 1)
            break;
        if (flag) {
            unknown[unknown.size() - 1].end_point = now_index - 1;
        }
        else {
            sample[sample.size() - 1].end_point = now_index - 1;
        }
        AnchorPoint anchor(last_index + 1, now_index, now_index, type);
        if (anchor.type == OUTER) {
            sample.push_back(anchor);
            flag = false;
        }
        else {
            unknown.push_back(anchor);
            flag = true;
        }
        last_index = now_index;
    }

    if (flag) {
        unknown.pop_back();
    }
    else {
        sample.pop_back();
    }
}

int StructurePropagation::getOneAnchorPos(int lastanchor_index, PointType &t, int curve_index,bool flag, vector<AnchorPoint>&unknown, vector<AnchorPoint>&sample){

    Point vertex = getLeftTopPoint(lastanchor_index, curve_index);
    Rect rec(vertex.x, vertex.y, block_size, block_size);

    int i = lastanchor_index+1;//this point will be the begin point in the next patch,so we should judge its type
    if (i >= pointlist[curve_index].size() - 1) {
        return pointlist[curve_index].size() - 1;
    }
    if (mask.at<uchar>(pointlist[curve_index][i]) == 0) {
        t = INNER;
    }
    else {
        t = OUTER;
    }

    while (i < pointlist[curve_index].size() && rec.contains(pointlist[curve_index][i])) {
        uchar tmp = mask.at<uchar>(pointlist[curve_index][i]);
        if (tmp == 0 && t == OUTER || tmp == 255 && t == INNER) {
            t = BORDER;
            if (flag) {
                int count = sample.size();
                if(count>0)
                    sample[count-1].type = BORDER;
            }
            else {
                int count = unknown.size();
                if (count>0)
                    unknown[count - 1].type = BORDER;
            }
        }
        i++;
    }
    return i;
}

Mat StructurePropagation::getOnePatch(Point p,Mat &img) {
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

Point StructurePropagation::getLeftTopPoint(int point_index, int curve_index) {
    Point p = pointlist[curve_index][point_index];
    int x = (p.x - block_size / 2) > 0 ? p.x - block_size / 2 : 0;
    int y = (p.y - block_size / 2) > 0 ? p.y - block_size / 2 : 0;
    return Point(x, y);
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

Rect StructurePropagation::getRect(Point p) {
    Point left_top = p - Point(block_size / 2, block_size / 2);
    Point right_down = left_top + Point(block_size, block_size);
    return Rect(left_top, right_down);
}

int sqr(int x)
{
    return x * x;
}

int dist(Vec3b V1, Vec3b V2)
{
    return sqr(int(V1[0]) - int(V2[0])) + sqr(int(V1[1]) - int(V2[1])) + sqr(int(V1[2]) - int(V2[2]));
    /*double pr = (V1[0] + V2[0]) * 0.5;
    return sqr(V1[0] - V2[0]) * (2 + (255 - pr) / 256)
    + sqr(V1[1] - V2[1]) * 4
    + sqr(V1[2] - V2[2]) * (2 + pr / 256);*/
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

        bool debug = false;
        bool debug2 = false;
        int k0 = min(x, bs), k1 = min(N - 1 - x, bs);
        int l0 = min(y, bs), l1 = min(M - 1 - y, bs);
        int sx, sy, min_diff = INT_MAX;
        for (int i = step; i + step < N; i += step)
            for (int j = step; j + step < M; j += step)
            {
                if (usable[i][j] == 2)continue;
                int tmp_diff = 0;
                for (int k = -k0; k <= k1; k++)
                    for (int l = -l0; l <= l1; l++)
                    {
                        //printf("%d %d %d %d %d %d\n", i + k, j + l, x + k, y + l, N, M);
                        if (my_mask[x + k][y + l] != 0)
                            tmp_diff += dist(result.at<Vec3b>(i + k, j + l), result.at<Vec3b>(x + k, y + l));
                    }
                sum_diff[i][j] = tmp_diff;
                if (min_diff > tmp_diff)
                {
                    sx = i;
                    sy = j;
                    min_diff = tmp_diff;
                }
            }


        if (debug)
        {
            printf("x = %d y = %d\n", x, y);
            printf("sx = %d sy = %d\n", sx, sy);
            printf("mindiff = %d\n", min_diff);
        }
        if (debug2)
        {
            match = result.clone();
        }
        for (int k = -k0; k <= k1; k++)
            for (int l = -l0; l <= l1; l++)
                if (my_mask[x + k][y + l] == 0)
                {
                    result.at<Vec3b>(x + k, y + l) = result.at<Vec3b>(sx + k, sy + l);
                    my_mask[x + k][y + l] = 1;
                    filled++;
                    if (debug)
                    {
                        result.at<Vec3b>(x + k, y + l) = Vec3b(255, 0, 0);
                        result.at<Vec3b>(sx + k, sy + l) = Vec3b(0, 255, 0);
                    }
                    if (debug2)
                    {
                        match.at<Vec3b>(x + k, y + l) = Vec3b(255, 0, 0);
                        match.at<Vec3b>(sx + k, sy + l) = Vec3b(0, 255, 0);
                    }
                }
                else
                {
                    if (debug)
                    {
                        printf("(%d,%d,%d) matches (%d,%d,%d)\n", result.at<Vec3b>(x + k, y + l)[0], result.at<Vec3b>(x + k, y + l)[1], result.at<Vec3b>(x + k, y + l)[2], result.at<Vec3b>(sx + k, sy + l)[0], result.at<Vec3b>(sx + k, sy + l)[1], result.at<Vec3b>(sx + k, sy + l)[2]);
                    }
                }
        if (debug) return;
        printf("done :%.2lf%%\n", 100.0 * filled / to_fill);
        imshow("run", result);
        waitKey(10);
    }
}