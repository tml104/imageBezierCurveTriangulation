/*
    º∆À„¡ΩÕºœÒRMSE Test
*/

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <CDT.h>
#include <Eigen/Dense>
#include <Eigen/LU>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <memory>
#include <random>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;


double rmse(cv::Mat& imageInput, cv::Mat& originImage) {

    double e = 0.0;

    std::cout << "hw:" << imageInput.size().height << " " << imageInput.size().width << std::endl;

    for (uint j = 0; j <= imageInput.size().height - 1; j++) { //row(y)
        for (uint i = 0; i <= imageInput.size().width - 1; i++) { //col(x)
            auto originalColor = originImage.at<cv::Vec3b>(j, i);
            auto interpolationResult = imageInput.at<cv::Vec3b>(j, i);


            for (uint channelId = 0; channelId < 3; channelId++) {
                double hx = originalColor[channelId] / 256.0;
                double px = interpolationResult[channelId] / 256.0;

                e += std::pow((hx - px), 2);
            }
        }
    }

    double rmse = std::sqrt(e / (imageInput.size().width * imageInput.size().height));

    return rmse;

}


int main()
{
    String im1 = "./[rasterization] imageResult4.jpg";
    String im2 = "./applefruit.jpg";


    cv::Mat im1Mat = imread(im1, IMREAD_COLOR);
    cv::Mat im2Mat = imread(im2, IMREAD_COLOR);

    

    std::cout << "rmse:" << rmse(im1Mat, im2Mat)<<std::endl;

    return 0;
}