#include<vector>

typedef enum {
    INNER,BORDER,OUTER
}PointType;

class AnchorPoint {

public:
    int begin_point;
    int anchor_point;
    int end_point;
    PointType type;
    int block_size;
    int curve_index;
    std::vector<int> neighbors;

    AnchorPoint(int anchor_point, int block_size, PointType type, int curve_index){
        this->anchor_point = anchor_point;
        this->begin_point = anchor_point - block_size / 2;
        this->end_point = anchor_point + block_size / 2;
        this->block_size = block_size;
        this->type = type;
        this->curve_index = curve_index;
    }
};
