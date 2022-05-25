#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "net.h"

using namespace std;

int main()
{
	cv::Mat img = cv::imread("test.jpg");
	int w = img.cols;
	int h = img.rows;
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, w, h, 224, 224);
	
	ncnn::Net net;
	net.load_param("resnet18.param");
	net.load_model("resnet18.bin");
	ncnn::Extractor ex = net.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(4);

	ncnn::Mat out;
	ex.input("x", in);
	ex.extract("y", out);

	ncnn::Mat out_flattened = out.reshape(out.w * out.h * out.c);
	vector<float> score;
	score.resize(out_flattened.w);
	for (int i = 0; i < out_flattened.w; ++i) {
		score[i] = out_flattened[i];
	}
	vector<float>::iterator max_id = max_element(score.begin(), score.end());
	printf("predicted class: %d, predicted value: %f", max_id - score.begin(), score[max_id - score.begin()]);

	net.clear();
	return 0;
}

