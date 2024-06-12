/*
    Just Test
*/

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <CDT.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <iostream>
#include <memory>
#include <random>
using namespace std;
using namespace cv;
using namespace cv::ximgproc;

cv::Mat func()
{
    cv::Mat ima(240, 320, CV_8UC3, cv::Scalar(0,0,255)); //Red
    return ima;
}

void salt(cv::Mat image, int n)
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> randomRow(0, image.rows - 1);
    std::uniform_int_distribution<int> randomCol(0, image.cols - 1);

    int i, j;
    for (int k = 0; k < n; k++) {
        i = randomCol(generator);
        j = randomRow(generator);

        // set white
        if (image.type() == CV_8UC1) {
            image.at<uchar>(j, i) = 255; 
        }
        else if(image.type() == CV_8UC3) {
            image.at<cv::Vec3b>(j, i)[0] = 255;
            image.at<cv::Vec3b>(j, i)[1] = 255;
            image.at<cv::Vec3b>(j, i)[2] = 255;

        }
    }
}

void colorReduce(cv::Mat image, int div = 64) {
    int nl = image.rows;
    int nc = image.cols * image.channels();
    for (int i = 0; i < nl; i++) {

        uchar* data = image.ptr<uchar>(i);
        for (int j = 0; j < nc; j++) {
            data[j] = data[j] / div * div + div / 2;
        }
    }
}

void shapen(const cv::Mat& image, cv::Mat& result) {
    result.create(image.size(), image.type());
    int nchannels = image.channels();
    for (int i = 1; i < image.rows - 1; i++) {
        const uchar* pre = image.ptr<const uchar>(i - 1);
        const uchar* cur = image.ptr<const uchar>(i);
        const uchar* nxt = image.ptr<const uchar>(i + 1);

        uchar* output = result.ptr<uchar>(i)+nchannels;
        for (int j = 1 * nchannels; j < (image.cols - 1) * nchannels; j++) {
            *(output)++ = cv::saturate_cast<uchar>(5 * cur[j] - cur[j - nchannels] - cur[j + nchannels] - pre[j] - nxt[j]);
        }

    }

    //result.row(0).setTo(cv::Scalar(0));
    //result.row(image.rows - 1).setTo(cv::Scalar(0));
    //result.col(0).setTo(cv::Scalar(0));
    //result.col(image.cols-1).setTo(cv::Scalar(0));
}

void wave(const cv::Mat& image, cv::Mat& result) {
    cv::Mat srcX(image.rows, image.cols, CV_32F);
    cv::Mat srcY(image.rows, image.cols, CV_32F);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            srcX.at<float>(i, j) = j;
            srcY.at<float>(i, j) = i+5*sin(j/10.0);
        }
    }

    cv::remap(image, result, srcX, srcY, cv::INTER_LINEAR);
}

/*
    OpenCV image test
*/
int main2()
{
    String imgPath = "./tml.jpg";

    Mat image;
    image = imread(imgPath, IMREAD_COLOR); // Read the file
    if (image.empty()) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // process

    cv::cvtColor(image, image, COLOR_BGR2GRAY);

    cout << "COLS:" << image.cols << endl;
    cout << "ROWS:" << image.rows << endl;
    cout << "CHANNELS:" << image.channels() << endl;

    //flip(image, image, 0); //0:v, 1:h, -1:both
    //circle(image, Point(720/2, 501/2), 30, 0, 3); //(720, 501)
    //putText(image, "7z", Point(720/2,501/2), FONT_HERSHEY_PLAIN, 2.0,255,2);

    //colorReduce(image);

    Mat res;
    //shapen(image, res);
    //wave(image, res);
    //floodFill(image, Point(720 / 2, 501 / 2), cv::Scalar(255, 255, 255), (cv::Rect*)0, cv::Scalar(35, 35, 35), cv::Scalar(35, 35, 35), cv::FLOODFILL_FIXED_RANGE);


    // end process

    //namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", image); // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}


struct CustomPoint2DForCDT
{
    double data[2];
};

struct CustomEdgeForCDT
{
    std::pair<std::size_t, std::size_t> vertices;
};

/*
    CDT Test
*/
int main3()
{
    std::vector<CustomPoint2DForCDT> points;
    std::vector<CustomEdgeForCDT> edges;

    //0~5
    points.push_back({ 0.0,0.0 });
    points.push_back({ 1.0,0.0 });
    points.push_back({ 1.0,1.0 });
    points.push_back({ 0.0,1.0 });
    points.push_back({ 0.2,0.5 });
    points.push_back({ 0.8,0.5 });

    //6~7
    points.push_back({ 0.5,0.2 });
    points.push_back({ 0.5,0.8 });


    //edges.push_back({ {4,5} });
    //edges.push_back({ {1,3} });
    //edges.push_back({ {0,2} });
    //edges.push_back({ {0,1} });
    //edges.push_back({ {1,2} });
    //edges.push_back({ {2,3} });
    //edges.push_back({ {3,0} });
    edges.push_back({ {4,5} });
    edges.push_back({ {6,7} });



    CDT::Triangulation<double> cdt;
    cdt.insertVertices(
        points.begin(),
        points.end(),
        [](const CustomPoint2DForCDT& p) { return p.data[0]; },
        [](const CustomPoint2DForCDT& p) { return p.data[1]; }
    );


    cdt.insertEdges(
        edges.begin(),
        edges.end(),
        [](const CustomEdgeForCDT& e) { return e.vertices.first; },
        [](const CustomEdgeForCDT& e) { return e.vertices.second; }
    );


    cdt.eraseSuperTriangle();
    //cdt.eraseOuterTriangles();

    auto cdtTri = cdt.triangles;
    auto cdtVer = cdt.vertices;
    auto cdtFix = cdt.fixedEdges;
    auto cdtExt = CDT::extractEdgesFromTriangles(cdt.triangles);


    cv::Mat edge_image_ed = Mat::zeros(500,800, CV_8UC3);
    for (auto e : cdtExt) {
        cout << "-- " << e.v1() << " " << e.v2() << " --" << endl;

        vector<Point> seg;
        seg.push_back({ 100+static_cast<int>(cdtVer[e.v1()].x * 100),100+static_cast<int>(cdtVer[e.v1()].y*100) });
        seg.push_back({ 100+static_cast<int>(cdtVer[e.v2()].x * 100),100+static_cast<int>(cdtVer[e.v2()].y*100) });

        cout << "  ++ " << seg[0].x << " " << seg[0].y <<" - " << seg[1].x << " " << seg[1].y << " ++" << endl;



        auto pts = &seg[0];
        int n = seg.size();
        polylines(edge_image_ed, &pts, &n, 1, false, Scalar((rand() & 255), (rand() & 255), (rand() & 255)), 1);
    }


    cout << edge_image_ed.size().height << "*" << edge_image_ed.size().width << endl; //500*800

    cv::imshow("3", edge_image_ed);
    waitKey(0);


    // add points

    //std::vector<CustomPoint2D> points2;

    //points2.push_back({ 0.5,0.2 });
    //
    //cdt.insertVertices(
    //    points2.begin(),
    //    points2.end(),
    //    [](const CustomPoint2D& p) { return p.data[0]; },
    //    [](const CustomPoint2D& p) { return p.data[1]; }
    //);

    //cdtExt = CDT::extractEdgesFromTriangles(cdt.triangles);
    //edge_image_ed = Mat::zeros(500, 800, CV_8UC3);
    //for (auto e : cdtExt) {
    //    cout << "-- " << e.v1() << " " << e.v2() << " --" << endl;

    //    vector<Point> seg;
    //    seg.push_back({ 100 + static_cast<int>(points[e.v1()].data[0] * 100),100 + static_cast<int>(points[e.v1()].data[1] * 100) });
    //    seg.push_back({ 100 + static_cast<int>(points[e.v2()].data[0] * 100),100 + static_cast<int>(points[e.v2()].data[1] * 100) });

    //    cout << "  ++ " << seg[0].x << " " << seg[0].y << " - " << seg[1].x << " " << seg[1].y << " ++" << endl;
    //    auto pts = &seg[0];
    //    int n = seg.size();
    //    polylines(edge_image_ed, &pts, &n, 1, false, Scalar((rand() & 255), (rand() & 255), (rand() & 255)), 1);
    //}
    //cv::imshow("3333", edge_image_ed);
    //waitKey(0);


    return 0;
}


cv::Point2f recursiveBezier(const std::vector<cv::Point2f>& control_points, float t)
{
    int n = control_points.size();
    if (n == 1) return control_points[0];
    std::vector<cv::Point2f> res_control_points;

    for (int i = 0; i < n - 1; i++) {
        res_control_points.push_back(cv::Point2f(
            (1 - t) * control_points[i].x + t * control_points[i + 1].x,
            (1 - t) * control_points[i].y + t * control_points[i + 1].y));
    }
    return recursiveBezier(res_control_points, t);

}

void bezier(const std::vector<cv::Point2f>& control_points, cv::Mat& window)
{
    for (double t = 0.0; t <= 1.0; t += 0.001)
    {
        cv::Point2f point = recursiveBezier(control_points, t);
        window.at<cv::Vec3b>(point.y, point.x) = cv::Vec3b(255,255,255);
    }
}

/*
    bezier curve test
*/
int main4() {
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));

    //std::shared_ptr<cv::Mat> t;
    //if (t == nullptr)
    //{
    //    cout << "NULLPTR" << endl;
    //    return 0;
    //}

    std::vector<cv::Point2f> control_points;

    //L1
    control_points.clear();
    control_points.push_back({ 2,2 });
    control_points.push_back({ 1,1 });
    control_points.push_back({ 1,3 });
    bezier(control_points, window);

    //L2
    control_points.clear();
    control_points.push_back({ 2,2 });
    control_points.push_back({ 5,3 });
    control_points.push_back({ 4,4 });
    bezier(control_points, window);

    //L3
    control_points.clear();
    control_points.push_back({ 1,3 });
    control_points.push_back({ 2,3 });
    control_points.push_back({ 4,4 });
    bezier(control_points, window);

    cv::floodFill(window, Point(2, 3), { 255,255,255 });

    cv::imshow("bz test", window);
    cv::waitKey(0);

    return 0;
}

/*
    Test for Linear Equations solver
*/
int main5() {
    Eigen::Matrix3d A {
        {5.0, 7.0, 3.0},
        {7.0, 10.0, 4.0},
        {3.0, 4.0, 1.0},
    };

    Eigen::Vector3d b{ {1.0, 2.0, 3.0 } };
    Eigen::Vector3d x;

    auto ALU = A.lu();

    x = ALU.solve(b);

    std::cout << "x:" << x << std::endl;
    std::cout << "d:" << ALU.determinant() << std::endl;
    std::cout << "x(0):" << x(0) << std::endl;
    std::cout << "x(1):" << x(1) << std::endl;
    std::cout << "x(2):" << x(2) << std::endl;


    b.setZero();
    std::cout << "b:" << b << std::endl;


    return 0;
}

int main(int argc, char** argv)
{
    return main2();
    //return main3();
    //return main4();
    //return main5();
}


// draw every triangle 
//cv::Mat tmpWindow2 = Mat::zeros(imageInput.size(), CV_8UC3);

//int _cnt = 0;
//for (auto& t : cdtTri) {
//    cv::Mat tmpWindow = Mat::zeros(imageInput.size()*10, CV_8UC3);

//    auto p1 = cv::Point(cdtVer[t.vertices[0]].x*10, cdtVer[t.vertices[0]].y*10);
//    auto p2 = cv::Point(cdtVer[t.vertices[1]].x*10, cdtVer[t.vertices[1]].y*10);
//    auto p3 = cv::Point(cdtVer[t.vertices[2]].x*10, cdtVer[t.vertices[2]].y*10);

//    auto p4 = (p1 + p2 + p3) / 3;

//    
//    // draw 3 lines
//    cv::line(tmpWindow, p1, p2, { 255,255,255 });
//    cv::line(tmpWindow, p2, p3, { 255,255,255 });
//    cv::line(tmpWindow, p3, p1, { 255,255,255 });
//    cv::floodFill(tmpWindow, p4, { 255,255,255 });


//    cv::resize(tmpWindow, tmpWindow, imageInput.size());

//    uint minx = std::min(cdtVer[t.vertices[0]].x, std::min(cdtVer[t.vertices[1]].x, cdtVer[t.vertices[2]].x));
//    uint maxx = std::max(cdtVer[t.vertices[0]].x, std::max(cdtVer[t.vertices[1]].x, cdtVer[t.vertices[2]].x));
//    uint miny = std::min(cdtVer[t.vertices[0]].y, std::min(cdtVer[t.vertices[1]].y, cdtVer[t.vertices[2]].y));
//    uint maxy = std::max(cdtVer[t.vertices[0]].y, std::max(cdtVer[t.vertices[1]].y, cdtVer[t.vertices[2]].y));


//    cv::Vec3i sumVec;
//    int sumcnt = 0;
//    for (uint j = miny; j <= maxy; j++) { //row
//        for (uint i = minx; i <= maxx; i++) { //col
//            if (tmpWindow.at<cv::Vec3b>(j, i) == cv::Vec3b(255, 255, 255))
//            {
//                sumVec += imageInput.at<cv::Vec3b>(j, i);
//                sumcnt++;
//            }
//        }
//    }
//    sumVec /= sumcnt;
//    for (uint j = miny; j <= maxy; j++) { //row
//        for (uint i = minx; i <= maxx; i++) { //col
//            if (tmpWindow.at<cv::Vec3b>(j, i) == cv::Vec3b(255, 255, 255))
//            {
//                tmpWindow2.at<cv::Vec3b>(j, i) = static_cast<cv::Vec3b>(sumVec);
//            }
//        }
//    }

//    cout << "[TriangulationRun] draw progress cnt:" <<++_cnt<<" / "<< cdtTri.size() << endl;
//}

//cv::imshow("[TriangulationRun] tmpWindow2", tmpWindow2);
//cv::waitKey(0);