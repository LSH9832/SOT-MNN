#ifndef SOT_PROCESS_H
#define SOT_PROCESS_H

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <math.h>


namespace sot {
    
    cv::Mat get_subwindow(cv::Mat _im, cv::Rect rect) {
        return _im(rect);
    }

    class Anchors {
        
        std::vector<float> ratios;
        std::vector<int> scales;
        int image_center=0;
        int size=0;
        
    public:
        std::vector<std::vector<float>> anchors, anchors_cxywh;
        float* anchors_xyxy;
        float* anchors_cxcywh;
        int anchor_num;
        int stride;

        Anchors();

        Anchors(YAML::Node ANCHOR, int _image_center=0, int _size=0) {
            stride = ANCHOR["STRIDE"].as<int>();
            ratios = ANCHOR["RATIOS"].as<std::vector<float>>();
            scales = ANCHOR["SCALES"].as<std::vector<int>>();

            image_center = _image_center;
            size = _size;
            anchor_num = scales.size() * ratios.size();

            generate_anchors();
        }

        Anchors(int _stride, std::vector<float> _ratios, std::vector<int> _scales, int _image_center=0, int _size=0) {
            stride = _stride;
            ratios = _ratios;
            scales = _scales;

            image_center = _image_center;
            size = _size;
            anchor_num = scales.size() * ratios.size();
        }

        void generate_anchors() {
            anchors.clear();

            int size = stride * stride;
            for (float r: ratios) {
                int ws = (int)sqrt((float)size / r);
                int hs = (int)sqrt(ws * r);

                for (int s: scales) {
                    std::vector<float> anchor;
                    anchor.push_back(-ws * s * 0.5);
                    anchor.push_back(-hs * s * 0.5);
                    anchor.push_back(ws * s * 0.5);
                    anchor.push_back(hs * s * 0.5);
                    anchors.push_back(anchor);

                    anchor.clear();
                    anchor.push_back(0);
                    anchor.push_back(0);
                    anchor.push_back(ws * s);
                    anchor.push_back(hs * s);

                    anchors_cxywh.push_back(anchor);
                }
            }
        }

        bool generate_all_anchors(int _img_center, int _size) {
            if (image_center == _img_center && size == _size) return false;

            image_center = _img_center;
            size = _size;

            int a0x = _img_center - _size / 2 * stride;

            float axyxy[anchors.size()][size][size][4];
            float acxcywh[anchors.size()][size][size][4];

            int count=0;
            for (std::vector<float> anchor: anchors) {
                float x1, y1, x2, y2;

                float cx = (anchor.at(0) + anchor.at(2)) / 2;
                float cy = (anchor.at(1) + anchor.at(3)) / 2;

                float w = (anchor.at(2) + anchor.at(0)) / 2;
                float h = (anchor.at(3) + anchor.at(1)) / 2;

                for (int i=0;i<size;i++) for (int j=0;j<size;j++) {
                    acxcywh[count][i][j][0] = cx + i * stride;
                    acxcywh[count][i][j][1] = cy + j * stride;
                    acxcywh[count][i][j][2] = w;
                    acxcywh[count][i][j][3] = h;

                    axyxy[count][i][j][0] = acxcywh[count][i][j][0] - w / 2;
                    axyxy[count][i][j][1] = acxcywh[count][i][j][1] - h / 2;
                    axyxy[count][i][j][2] = axyxy[count][i][j][0] + w;
                    axyxy[count][i][j][3] = axyxy[count][i][j][1] + h;
                }

                count++;
            }

            anchors_xyxy = &axyxy[0][0][0][0];
            anchors_cxcywh = &acxcywh[0][0][0][0];

            return true;
        }

    };
    
}







#endif
#define SOT_PROCESS_H