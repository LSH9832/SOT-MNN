#include <opencv2/opencv.hpp>
#include "image_utils/mnn.h"
#include "argparse/argparser.h"
#include "print_utils.h"
#include <sys/time.h>

argsutil::argparser get_args(int argc, char** argv) {
    auto parser = argsutil::argparser("mnn sot parser");

    parser.add_option<std::string>("-c", "--cfg", "config file path", "config/alex.yaml");
    parser.add_option<std::string>("-s", "--source", "video source", "/dev/video0");
    parser.add_option<bool>("-p", "--pause-at-start", "pause at start", false);
    parser.add_help_option();
    parser.parse(argc, argv);

    return parser;
}


int main(int argc, char** argv) {

    auto args = get_args(argc, argv);
    auto tracker = mnn_sot::SOTracker(args.get_option_string("--cfg"));
    
    // auto cap = cv::VideoCapture("/home/lsh/Videos/test.avi");
    // auto cap = cv::VideoCapture("/home/lsh/Videos/cloud.mp4");
    auto cap = cv::VideoCapture(args.get_option_string("--source"));

    bool first=true;

    int delay=args.get_option_bool("--pause-at-start")?0:1;

    timeval t0, t1;

    cv::Mat image;
    while (cap.isOpened()) {
        if (!cap.read(image)) break;
        if (image.empty()) break;

        if (first) {
            first = false;
            auto bbox = cv::selectROI("select roi", image, true, false);
            tracker.init(image.clone(), bbox);
            cv::destroyWindow("select roi");
        }
        else {
            
            gettimeofday(&t0, NULL);
            auto result = tracker.track(image.clone());
            gettimeofday(&t1, NULL);

            INFO << "track latency: " << get_time_interval(t0, t1) 
                 << "ms, confidence: " << result.prob << ENDL;

            cv::rectangle(image, result.location, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::imshow("result", image);
            int k = cv::waitKey(delay);
            if (k==27) break;
            else if (k==(int)(' ')) delay = 1 - delay;
            else if (k==(int)('s')) cv::imwrite("demo.jpg", image);
        }
        
    }

    cv::destroyAllWindows();

    return 0;
    // auto cap = cv::VideoCapture("/home/lsh/Videos/test.avi")
}