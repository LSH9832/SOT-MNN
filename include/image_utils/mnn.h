#ifndef MNN_H
#define MNN_H

#include <thread>

#include <opencv2/opencv.hpp>

#include "MNN/MNNDefine.h"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/AutoTime.hpp"
#include "MNN/Interpreter.hpp"

#include "image_utils/detect_process.h"
#include "image_utils/sot_process.h"
#include "yaml-cpp/yaml.h"

#include "print_utils.h"
#include <math.h>

// #include <time.h>
// #include <ctime>
#include <sys/time.h>


namespace mnn_det {

    float img2mnn(cv::Mat &img, MNN::Tensor *input_tensor, int batch_id=0, bool require_size_equal=false) {
        assert(batch_id < input_tensor->shape().at(0));
        assert(input_tensor->shape().size() == 4);

        cv::Size fixed_size(input_tensor->shape().at(3), input_tensor->shape().at(2));
        detect::resizeInfo resize_info;

        if (require_size_equal) {
            // INFO << input_tensor->shape().at(0) << " " << input_tensor->shape().at(1) << " " << input_tensor->shape().at(2) << " " << input_tensor->shape().at(3) << ENDL;
            // INFO << fixed_size << ENDL;
            // INFO << img.size() << ENDL;
            // if (fixed_size != img.size()) {
            //     WARN << fixed_size << ENDL;
            //     WARN << img.size() << ENDL;
            // }
            // INFO << fixed_size << ENDL;
            // INFO << img.size() << ENDL;
            assert(fixed_size == img.size());
            resize_info.resized_img = img.clone();
            resize_info.factor = 1.;
        }
        else resize_info = detect::resizeAndPad(img, fixed_size, false, false);

        int img_C3 = resize_info.resized_img.channels();
        auto nchwTensor = new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE);

        int fixedlw = fixed_size.height * fixed_size.width;
        int fixedchw = img_C3 * fixedlw;
        for (int i = 0; i < fixed_size.height; i++)
            for (int j = 0; j < fixed_size.width; j++)
                for (int k = 0; k < img_C3; k++) {
                    nchwTensor->host<float>()[batch_id * fixedchw + k * fixedlw + i * fixed_size.width + j] = resize_info.resized_img.at<cv::Vec3b>(i, j)[k];
                }

        input_tensor->copyFromHostTensor(nchwTensor);
        delete nchwTensor;

        return resize_info.factor;
    }

    class YOLO {

        std::shared_ptr<MNN::Interpreter> net; //模型翻译创建
        MNN::ScheduleConfig config;            //计划配置
        MNN::Session *session;

        MNN::Tensor *inputTensor;
        MNN::Tensor *outputTensor;

        cv::Size input_size, img_size;
        int num_batch;
        int num_dets;
        int length_array;

        float *preds;

        std::vector<float> batch_ratios;
        std::vector<std::vector<detect::Object>> batch_objects;

        timeval preprocess_start, infer_start, postprocess_start, total_end;

    public:
        std::string mnn_model;
        std::vector<std::string> names;
        int num_threads=4;

        float conf_thres=0.25;
        float nms_thres=0.45;

        bool config_set=false;
        bool model_loaded=false;

        cv::Mat *current_img;
        float current_ratio;
        MNNForwardType DEVICE=MNN_FORWARD_AUTO;

        YOLO() {};

        ~YOLO() {
            INFO << "release yolo" << ENDL;
            // delete session;
            // delete inputTensor;
            // delete outputTensor;
            // delete current_img;
            // delete preds;
        }

        YOLO(std::string mnn_config) {
            set_config(mnn_config);
            // INFO << config_set << ENDL;
        }

        void set_config(std::string mnn_config) {
            YAML::Node cfg = YAML::LoadFile(mnn_config);
            mnn_model = cfg["model"].as<std::string>();
            names = cfg["names"].as<std::vector<std::string>>();
            num_threads = cfg["num_threads"].as<int>();
            conf_thres = cfg["conf_thres"].as<float>();
            nms_thres = cfg["nms_thres"].as<float>();
            config_set = true;
        }

        void set_device(bool _cpu=false, bool _gpu=false) {
            if (_cpu) DEVICE = MNN_FORWARD_CPU;
            if (_gpu) DEVICE = MNN_FORWARD_CUDA;
        }

        void set_confidence_threshold(float _value) {
            conf_thres = _value;
        }

        void set_nms_threshold(float _value) {
            nms_thres = _value;
        }

        bool load_model() {
            if (!config_set) return false;

            net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(this->mnn_model.c_str()));
            config.numThread = num_threads;

            // config.type = MNN_FORWARD_AUTO;
            config.type = DEVICE;

            session = net->createSession(config);
            //获取输入输出tensor
            inputTensor = net->getSessionInput(session, NULL);
            outputTensor = net->getSessionOutput(session, NULL);

            length_array = outputTensor->shape().at(2);

            assert(inputTensor->shape().size() == 4);    // batch, c, h, w
            assert(outputTensor->shape().size() == 3);   // batch, num_dets, array
            assert(length_array == names.size()+5);
            
            num_batch = inputTensor->shape().at(0);
            input_size.height = inputTensor->shape().at(2);
            input_size.width = inputTensor->shape().at(3);
            num_dets = outputTensor->shape().at(1);

            model_loaded=true;

            return model_loaded;
        }

        void _forward(cv::Mat *img, int batch_id=0) {
            
            gettimeofday(&preprocess_start, NULL);

            current_img = img;
            img_size = current_img->size();
            current_ratio = img2mnn(*current_img, inputTensor, batch_id);

            gettimeofday(&infer_start, NULL);

            net->runSession(session);

            auto outNchwTensor = new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE);
            outputTensor->copyToHostTensor(outNchwTensor);
            preds = outNchwTensor->host<float>();
            // INFO << "RATIO: " << current_ratio << ENDL;

            gettimeofday(&postprocess_start, NULL);
        }

        void __preprocess_batch_once(cv::Mat img, int batch_id=0) {
            img_size = img.size();
            batch_ratios[batch_id] = img2mnn(img, inputTensor, batch_id);
        }

        void _forward_multi(std::vector<cv::Mat> imgs) {
            assert(imgs.size()==num_batch);
            batch_ratios.clear();
            batch_ratios.resize(num_batch);

            std::vector<std::thread> threads;
            for (int i=0;i<num_batch;i++) {
                std::thread _t(std::mem_fn(&YOLO::__preprocess_batch_once), this, imgs.at(i), i);
                threads.push_back(std::move(_t));
            }
            for (auto& _t: threads) _t.join();

            net->runSession(session);

            auto outNchwTensor = new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE);
            outputTensor->copyToHostTensor(outNchwTensor);
            preds = outNchwTensor->host<float>();

            // INFO << preds << ENDL;
        }

        void generate_yolo_proposals(std::vector<detect::Object>& objects, int batch_id) {
            for (int anchor_idx = 0; anchor_idx < num_dets; anchor_idx++) {
                const int basic_pos = batch_id * num_dets + anchor_idx * length_array;
                
                // 解析类别及其置信度
                int label = -1;
                float prob = 0.0;
                // INFO << 1 << ENDL;
                float box_objectness = preds[basic_pos+4];    // obj conf
                // INFO << 2 << ENDL;
                for (int class_idx = 0; class_idx < length_array - 5; class_idx++)
                {
                    float box_cls_score = preds[basic_pos + 5 + class_idx];
                    float box_prob = box_objectness * box_cls_score;
                    if (box_prob > conf_thres && box_prob > prob) {
                        label = class_idx;
                        prob = box_prob;
                    }
                }
                // INFO << 3 << ENDL;

                // 若置信度大于阈值则输出
                if(prob > conf_thres) {
                    detect::Object obj;
                    obj.rect.width = preds[basic_pos+2];
                    obj.rect.height = preds[basic_pos+3];
                    obj.rect.x = preds[basic_pos+0] - obj.rect.width * 0.5f;
                    obj.rect.y = preds[basic_pos+1] - obj.rect.height * 0.5f;
                    
                    obj.label = label;
                    obj.prob = prob;

                    objects.push_back(obj);
                }
                // INFO << 4 << ENDL;
            }
        }

        void _decode_outputs(int batch_id=0) {
            batch_objects[batch_id] = this->_decode_output(batch_id, batch_ratios[batch_id]);
        }

        std::vector<detect::Object> _decode_output(int batch_id=0, float ratio=1.0) {
            std::vector<detect::Object> proposals, objects;
            std::vector<int> picked;

            this->generate_yolo_proposals(proposals, batch_id);
            detect::qsort_descent_inplace(proposals);
            detect::nms_sorted_bboxes(proposals, picked, nms_thres);

            int count = picked.size();
            objects.resize(count);
            for (int i = 0; i < count; i++)
            {
                objects[i] = proposals[picked[i]];

                float x0 = (objects[i].rect.x) * ratio;
                float y0 = (objects[i].rect.y) * ratio;
                float x1 = (objects[i].rect.x + objects[i].rect.width) * ratio;
                float y1 = (objects[i].rect.y + objects[i].rect.height) * ratio;

                // clip
                x0 = std::max(std::min(x0, (float)(img_size.width - 1)), 0.f);
                y0 = std::max(std::min(y0, (float)(img_size.height - 1)), 0.f);
                x1 = std::max(std::min(x1, (float)(img_size.width - 1)), 0.f);
                y1 = std::max(std::min(y1, (float)(img_size.height - 1)), 0.f);

                objects[i].rect.x = x0;
                objects[i].rect.y = y0;
                objects[i].rect.width = x1 - x0;
                objects[i].rect.height = y1 - y0;
            }

            return objects;

        }

        std::vector<std::vector<detect::Object>> batch_infer(std::vector<cv::Mat> images) {
            this->_forward_multi(images);
            batch_objects.clear();
            batch_objects.resize(num_batch);
            std::vector<std::thread> threads;
            for (int i=0;i<num_batch;i++) {
                std::thread _t(std::mem_fn(&YOLO::_decode_outputs), this, i);
                threads.push_back(std::move(_t));
            }
            for (auto& _t: threads) _t.join();

            return batch_objects;
        }

        void show_msg(int obj_num) {
            INFO << "pre-process: " << get_time_interval(preprocess_start, infer_start)
                 << "ms, inference: " << get_time_interval(infer_start, postprocess_start)
                 << "ms, post-process: " << get_time_interval(postprocess_start, total_end)
                 << "ms, num objects:" << obj_num << ENDL;
        }

        std::vector<detect::Object> infer(cv::Mat image) {
            this->_forward(&image, 0);
            std::vector<detect::Object> results = this->_decode_output(0, current_ratio);
            gettimeofday(&total_end, NULL);
            show_msg(results.size());
            return results;
        }


    };

}


namespace mnn_sot {

    struct TrackInfo
    {
        float prob=0.;
        cv::Rect location;
    };

    std::vector<float> hanning(int M) {
        std::vector<float> ret;
        for (int i=0;i<M;i++) {
            ret.push_back((1.-cos(2 * M_PI * i / (M - 1)))/2);
        }
        return ret;
    }

    std::vector<std::vector<float>> outer(std::vector<float> _a, std::vector<float> _b) {
        std::vector<std::vector<float>> ret;
        for (float i: _a) {
            std::vector<float> _r;
            for (float j: _b) _r.push_back(i * j);
            ret.push_back(_r);
        }
        return ret;
    }

    std::vector<float> tile_expand(std::vector<std::vector<float>> _input, int times) {
        std::vector<float> ret;
        for(int i=0;i<times;i++)
            for (std::vector<float> anchor: _input)
                for (float _a: anchor)
                    ret.push_back(_a);
        return ret;
    }

    cv::Mat get_subwindow(const cv::Mat& im, const cv::Point& pos, const cv::Size& model_sz, const cv::Size& original_sz, const cv::Scalar& avg_chans) {  
        
        // INFO << "input image size : " << im.size() << ENDL;
        // INFO << "position         : " << pos << ENDL;
        // INFO << "model size       : " << model_sz << ENDL;
        // INFO << "ori size         : " << original_sz << ENDL;



        cv::Size sz = original_sz;  
        cv::Size im_sz = im.size();  
        double c = (original_sz.width + 1) / 2.0;  
        int context_xmin = std::floor(pos.x - c + 0.5);  
        int context_xmax = context_xmin + sz.width - 1;  
        int context_ymin = std::floor(pos.y - c + 0.5);  
        int context_ymax = context_ymin + sz.height - 1;

        int left_pad = std::max(0, -context_xmin);  
        int top_pad = std::max(0, -context_ymin);  
        int right_pad = std::max(0, context_xmax - im_sz.width + 1);  
        int bottom_pad = std::max(0, context_ymax - im_sz.height + 1);  
    
        context_xmin += left_pad;  
        context_xmax -= right_pad;  
        context_ymin += top_pad;  
        context_ymax -= bottom_pad;  
    
        int rows = im.rows;  
        int cols = im.cols;  
        cv::Mat im_patch; // 声明im_patch  
        if (top_pad || bottom_pad || left_pad || right_pad) {  

            // WARN << "NEED PAD" << top_pad << bottom_pad << left_pad << right_pad <<  ENDL;

            im_patch = im(cv::Rect(context_xmin, context_ymin, context_xmax-context_xmin, context_ymax-context_ymin)).clone();
            cv::copyMakeBorder(im_patch, im_patch, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, avg_chans);
            

        } else {  
            // INFO << im.size() << " " << im.rows << ENDL;
            // INFO << context_xmin << " " << context_ymin << " " << sz.width << " " << sz.height << ENDL;
            // WARN << "NO NEED PAD" << ENDL;
            im_patch = im(cv::Rect(context_xmin, context_ymin, sz.width, sz.height)).clone(); // 在这里复制图像子窗口
        }

        // WARN << im_patch.size() << ENDL;

        

        // cv::imshow("im_patch", im_patch);
        // cv::waitKey(0);

        if (!im_patch.empty() && (model_sz != im_patch.size())) { // 在这里进行非空判断和大小调整  
            cv::resize(im_patch, im_patch, model_sz);  
        } 

        

        // WARN << im_patch.size() << ENDL;

        return im_patch;  
    }

    class SOTTemplateModel {
        std::shared_ptr<MNN::Interpreter> net; //模型翻译创建
        MNN::ScheduleConfig config;            //计划配置
        MNN::Session *session;

        MNN::Tensor *inputTensor;
        std::vector<MNN::Tensor*> outputTensors;
        

    public:

        std::vector<MNN::Tensor*> returnTensors;
        bool model_loaded=false;
        
        int num_threads=5;

        SOTTemplateModel() {};
        SOTTemplateModel(std::string model_file, 
                         std::vector<std::string> output_names,
                         MNNForwardType DEVICE=MNN_FORWARD_AUTO) {

            INFO << "LOADING TEMPLATE MODEL: " << model_file << ENDL;

            net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
            config.numThread = num_threads;
            config.type = DEVICE;
            session = net->createSession(config);
            
            //获取输入输出tensor
            inputTensor = net->getSessionInput(session, NULL);

            INFO << "INPUT SHAPE: [ ";
            for (int s: inputTensor->shape()) std::cout << s << " ";
            std::cout << "]" << ENDL;

            outputTensors.resize(output_names.size());
            for (int i=0;i<output_names.size();i++) {
                outputTensors[i] = net->getSessionOutput(session, output_names[i].c_str());
                INFO << "OUTPUT-" << output_names[i] << " SHAPE: [ ";
                for (int s: outputTensors[i]->shape()) std::cout << s << " ";
                std::cout << "]" << ENDL;
            }

            model_loaded=true;
        }

        void forward_only(cv::Mat _img) {
            if (!model_loaded) {
                ERROR << "SOT Template Model not loaded!" << ENDL;
                return;
            }
            mnn_det::img2mnn(_img, inputTensor, 0, true);
            net->runSession(session);
            
            // for (auto returnTensor: returnTensors) {
            //     delete returnTensor;
            //     returnTensor = NULL;
            // }
            returnTensors.clear();
            for (int i=0;i<outputTensors.size();i++) {
                auto outNchwTensor = new MNN::Tensor(outputTensors[i], MNN::Tensor::CAFFE);
                outputTensors[i]->copyToHostTensor(outNchwTensor);
                returnTensors.push_back(outNchwTensor);
                // delete outNchwTensor;
            }
        }
        
        std::vector<MNN::Tensor*> result() {
            return returnTensors;
        }

        std::vector<float*> forward(cv::Mat _img) {
            std::vector<float*> preds;
            if (!model_loaded) {
                ERROR << "SOT Template Model not loaded!" << ENDL;
                return preds;
            }

            mnn_det::img2mnn(_img, inputTensor, 0, true);

            net->runSession(session);

            
            for (auto outputTensor: outputTensors) {
                auto outNchwTensor = new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE);
                outputTensor->copyToHostTensor(outNchwTensor);
                preds.push_back(outNchwTensor->host<float>());
                // delete outNchwTensor;
            }
            
            return preds;
        }

        std::vector<MNN::Tensor*> forward_tensor(cv::Mat _img) {
            std::vector<MNN::Tensor*> outNchwTensors;
            if (!model_loaded) {
                ERROR << "SOT Template Model not loaded!" << ENDL;
                return outNchwTensors;
            }

            mnn_det::img2mnn(_img, inputTensor, 0, true);

            net->runSession(session);

            for (int i=0;i<outputTensors.size();i++) {
                auto outNchwTensor = new MNN::Tensor(outputTensors[i], MNN::Tensor::CAFFE);
                outputTensors[i]->copyToHostTensor(outNchwTensor);
                outNchwTensors.push_back(outNchwTensor);
                // delete outNchwTensor;
            }
            return outNchwTensors;
        }
    };

    class SOTTrackModel {
        std::shared_ptr<MNN::Interpreter> net; //模型翻译创建
        MNN::ScheduleConfig config;            //计划配置
        MNN::Session *session;

        std::vector<MNN::Tensor*> inputTensors;
        std::vector<MNN::Tensor*> outputTensors;

        std::vector<MNN::Tensor*> templates;

        
        // MNNForwardType DEVICE=MNN_FORWARD_CPU;

        
    
    public:
        int num_preds;
        int num_threads=5;
        bool model_loaded=false;
        bool template_set=false;
        
        SOTTrackModel() {};
        SOTTrackModel(std::string model_file, 
                      std::vector<std::string> input_names, 
                      std::vector<std::string> output_names,
                      MNNForwardType DEVICE=MNN_FORWARD_AUTO) {
            INFO << "LOADING TRACK MODEL: " << model_file << ENDL;
            net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
            config.numThread = num_threads;
            config.type = DEVICE;
            session = net->createSession(config);
            
            //获取输入输出tensor
            inputTensors.resize(input_names.size());
            for (int i=0;i<input_names.size();i++) {
                inputTensors[i] = net->getSessionInput(session, input_names[i].c_str());
                INFO << "INPUT-" << input_names[i] << " SHAPE: [ ";
                for (int s: inputTensors[i]->shape()) std::cout << s << " ";
                std::cout << "]" << ENDL;
            }
            
            outputTensors.resize(output_names.size());
            for (int i=0;i<output_names.size();i++) {
                outputTensors[i] = net->getSessionOutput(session, output_names[i].c_str());
                INFO << "OUTPUT-" << output_names[i] << " SHAPE: [ ";
                for (int s: outputTensors[i]->shape()) std::cout << s << " ";
                std::cout << "]" << ENDL;
            }
            
            num_preds = outputTensors[0]->shape().at(0);   // 0: score [num_preds], 1, pred_bbox [4, num_preds]

            INFO << "NUM PREDS: " << num_preds << ENDL;

            model_loaded=true;
        }

        void set_template(std::vector<MNN::Tensor*> _templates) {
            templates = _templates;
            template_set=true;
        }

        std::vector<float*> forward(cv::Mat _img) {
            std::vector<float*> preds;

            if (!model_loaded) {
                ERROR << "SOT Track Model not loaded!" << ENDL;
                return preds;
            }

            if (!template_set) {
                ERROR << "SOT Template not set!" << ENDL;
                return preds;
            }

            mnn_det::img2mnn(_img, inputTensors[0], 0, true);

            for (int i=0;i<templates.size();i++) {
                inputTensors[i+1]->copyFromHostTensor(templates[i]);
            }

            net->runSession(session);

            for (auto outputTensor: outputTensors) {
                auto outNchwTensor = new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE);
                outputTensor->copyToHostTensor(outNchwTensor);
                preds.push_back(outNchwTensor->host<float>());
                // delete outNchwTensor;
            }
            
            return preds;
        }


    };

    class SOTracker {
        YAML::Node cfg, model_cfg;
        int score_size;
        int anchor_num;

        float content_amount;

        cv::Point center_pose;
        cv::Size size, track_exemplar_size;

        SOTTemplateModel template_model;
        SOTTrackModel track_model;

        TrackInfo track_result;

        int w_add_h;
        float w_z;
        float h_z;
        int track_instance_size;

        float WINDOW_INFLUENCE;
        float PENALTY_K;
        float LR;

        bool tracker_init=false;
        cv::Scalar mean_color;

        std::vector<float> window;

        float _change(float r) {
            return MAX(r, 1./r);
        }

        float _sz(float w, float h) {
            float pad = (w + h) / 2;
            return sqrt((w + pad) * (h + pad));
        }

        cv::Rect _bbox_clip(float cx, float cy, float width, float height, cv::Size _ori_size) {
            cx = MAX(0, MIN(cx, _ori_size.width));
            cy = MAX(0, MIN(cy, _ori_size.height));
            width = MAX(10, MIN(width, _ori_size.width));
            height = MAX(10, MIN(height, _ori_size.height));
            return cv::Rect((int)cx, (int)cy, (int)width, (int)height);
        }

    public:


        SOTracker() {};
        SOTracker(std::string config) {
            cfg = YAML::LoadFile(config);

            std::string mode = cfg["mode"].as<std::string>();
            MNNForwardType DEVICE=MNN_FORWARD_AUTO;

            if (mode == "A" || mode == "a") DEVICE = MNN_FORWARD_AUTO;
            else if (mode == "C" || mode == "c") DEVICE = MNN_FORWARD_CPU;
            else if (mode == "G" || mode == "g") DEVICE = MNN_FORWARD_CUDA;
                
            template_model = SOTTemplateModel(cfg["template_model_file_path"].as<std::string>(), 
                                              cfg["template_model_output_names"].as<std::vector<std::string>>(),
                                              DEVICE);

            track_model = SOTTrackModel(cfg["track_model_file_path"].as<std::string>(),
                                        cfg["track_model_input_names"].as<std::vector<std::string>>(),
                                        cfg["track_model_output_names"].as<std::vector<std::string>>(),
                                        DEVICE);

            
            

            model_cfg = YAML::LoadFile(cfg["model_config"].as<std::string>());
            score_size = (model_cfg["TRACK"]["INSTANCE_SIZE"].as<int>()-
                          model_cfg["TRACK"]["EXEMPLAR_SIZE"].as<int>()) / 
                          model_cfg["ANCHOR"]["STRIDE"].as<int>() + 1 + 
                          model_cfg["TRACK"]["BASE_SIZE"].as<int>();
            
            // self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
            int num_anchors = model_cfg["ANCHOR"]["RATIOS"].as<std::vector<float>>().size() * 
                              model_cfg["ANCHOR"]["SCALES"].as<std::vector<int>>().size();
            
            std::vector<float> h = hanning(score_size);
            window = tile_expand(outer(h, h), num_anchors);
            track_instance_size = model_cfg["TRACK"]["INSTANCE_SIZE"].as<int>();
            content_amount = model_cfg["TRACK"]["CONTEXT_AMOUNT"].as<float>();
            WINDOW_INFLUENCE = model_cfg["TRACK"]["WINDOW_INFLUENCE"].as<float>();
            PENALTY_K = model_cfg["TRACK"]["PENALTY_K"].as<float>();
            LR = model_cfg["TRACK"]["LR"].as<float>();

            int _sz = model_cfg["TRACK"]["EXEMPLAR_SIZE"].as<int>();
            track_exemplar_size.width = _sz;
            track_exemplar_size.height = _sz;

        }

        void init(cv::Mat _img, cv::Rect _bbox) {

            center_pose.x = _bbox.x + (_bbox.width-1) / 2;
            center_pose.y = _bbox.y + (_bbox.height-1) / 2;
            size.width = _bbox.width;
            size.height = _bbox.height;

            w_add_h = size.width + size.height;
            w_z = content_amount * w_add_h + size.width;
            h_z = content_amount * w_add_h + size.height;
            int s_z = (int)round(sqrt(w_z * h_z));

            mean_color = cv::mean(_img);

            cv::Mat z_crop = get_subwindow(_img, center_pose, track_exemplar_size, cv::Size(s_z, s_z), mean_color);

            template_model.forward_only(z_crop);
            track_model.set_template(template_model.result());
            tracker_init = true;
        }

        TrackInfo track(cv::Mat _img) {

            w_add_h = size.width + size.height;
            w_z = content_amount * w_add_h + size.width;
            h_z = content_amount * w_add_h + size.height;
            float s_z = round(sqrt(w_z * h_z));
            
            float s_x = s_z * ((float)track_instance_size / (float)track_exemplar_size.width);

            // float scale_z = (float)track_exemplar_size.width / s_z;
            float scale_z = (float)track_instance_size / s_x;

            cv::Mat x_crop = get_subwindow(_img.clone(), center_pose,
                                           cv::Size(track_instance_size, track_instance_size),
                                           cv::Size((int)round(s_x), (int)round(s_x)), 
                                           mean_color);
            // INFO << 1 << ENDL;
            std::vector<float*> preds = track_model.forward(x_crop);
            // INFO << 2 << ENDL;
            float* score = preds[0];
            float* pred_bbox = preds[1];


            float max_pscore=0.;
            float lr_ratio=0.;
            int max_idx=0;
            for (int i=0;i<track_model.num_preds;i++) {
                // float x = pred_bbox[0 * track_model.num_preds + i] / scale_z;
                // float y = pred_bbox[1 * track_model.num_preds + i] / scale_z;
                float w = pred_bbox[2 * track_model.num_preds + i];
                float h = pred_bbox[3 * track_model.num_preds + i];
                
                float this_score = score[i];

                float s_c = _change(_sz(w, h) / _sz(scale_z*size.width, scale_z*size.height));
                float r_c = _change(((float)size.width/(float)size.height)/(w/h));


                float penalty = exp(-(r_c * s_c - 1) * PENALTY_K);

                float pscore = penalty * this_score;

                pscore = pscore * (1 - WINDOW_INFLUENCE) + window[i] * WINDOW_INFLUENCE;

                if (i==0 || pscore > max_pscore) {
                    max_pscore = pscore;
                    lr_ratio = penalty * this_score;
                    max_idx = i;
                }
            }

            float x = pred_bbox[0 * track_model.num_preds + max_idx] / scale_z;
            float y = pred_bbox[1 * track_model.num_preds + max_idx] / scale_z;
            float width = pred_bbox[2 * track_model.num_preds + max_idx] / scale_z;
            float height = pred_bbox[3 * track_model.num_preds + max_idx] / scale_z;

            float lr = lr_ratio * LR;

            float cx = x + center_pose.x;
            float cy = y + center_pose.y;

            // smooth bbox
            // lr = 1.;
            // INFO << "MAX SCORE: " << max_pscore << ENDL;
            // INFO << width << "," << height << ENDL;
            width = lr * width + (1. - lr) * size.width;
            height = lr * height + (1. - lr) * size.height;

            // clip boundary
            cv::Rect rect = _bbox_clip(cx, cy, width, height, _img.size());

            // update state
            center_pose.x = rect.x;
            center_pose.y = rect.y;
            size.width = rect.width;
            size.height = rect.height;

            track_result.location.x = rect.x - rect.width / 2;
            track_result.location.y = rect.y - rect.height / 2;
            track_result.location.width = rect.width;
            track_result.location.height = rect.height;

            track_result.prob = score[max_idx];

            // INFO << "PROB: " << track_result.prob << ENDL;
            return track_result;
        }

    };
}



// void 

#endif
#define MNN_H