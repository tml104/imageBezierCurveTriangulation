#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <CDT.h>

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

const double MAXDIS = 99999.0;

struct CustomPoint2DForCDT
{
    cv::Point data;
};

struct CustomEdgeForCDT
{
    std::pair<std::size_t, std::size_t> vertices;
};

struct TriangulationMesh {
    struct CurvedTriangle {
        std::vector<uint> verticesIndexList;
        std::vector<uint> incidentTrianglesIndexList;
        std::vector<std::pair<uint, uint>> pixelsList;

        // TODO: interpolation parameters...
        // now only const

        cv::Vec3i constColor;
    };

    struct Edge {
        std::pair<uint,uint> vertexIndexPair;
        std::pair<double, double> controlPoint;
        //std::vector<std::pair<uint, uint>> bezierPointsList;
        std::shared_ptr<CurvedTriangle> leftTriangle;
        std::shared_ptr<CurvedTriangle> rightTriangle;

        void setVertexIndex(const std::pair<uint,uint> &p) {
            vertexIndexPair = p;
        }

        void setControlPoint(const std::pair<double,double> &p1, const std::pair<double, double> &p2) {
            controlPoint = { (p1.first + p2.first) / 2.0, (p1.second + p2.second) / 2.0 };
        }

        void setControlPoint(cv::Point2d cvc_new) {
            controlPoint = { cvc_new.x, cvc_new.y };
        }

        cv::Point2d getControlPointCV() {
            return cv::Point2d(controlPoint.first, controlPoint.second);
        }

        void setTrianglePtr(std::shared_ptr<CurvedTriangle> ptr, bool isRight = false) {
            if (isRight) {
                rightTriangle = ptr;
                return;
            }
            leftTriangle = ptr;
        }
    };

    struct Vertex {
        std::pair<double, double> coordinate;
        std::vector<std::shared_ptr<Edge>> incidentEdgesList;
        std::vector<uint> incidentVerticesList;

        void setCoordinate(double x, double y) {
            coordinate = { x,y };
        }
    };

    std::vector<std::shared_ptr<Vertex>> pointsList;
    std::map<std::pair<uint, uint>, std::shared_ptr<Edge>> edgesMap;
    std::vector<std::shared_ptr<CurvedTriangle>> trianglesList;

    TriangulationMesh() {}

    TriangulationMesh(CDT::Triangulation<double> &cdt) {

        _setPointsList(cdt); 
        _setEdgesMap(cdt); 
        _setTrianglesList(cdt);
        _setPointsIncidentVertices();
    }

    void _setPointsList(CDT::Triangulation<double>& cdt) {

        auto cdtVer = cdt.vertices;
        cout << "[TriangulationMesh] cdtVer size:" << cdtVer.size() << endl;

        for (auto& v2d : cdtVer) {

            auto vertexPtr = std::make_shared<Vertex>();
            vertexPtr->setCoordinate(v2d.x, v2d.y);

            this->pointsList.emplace_back(vertexPtr);
        }
    }

    void _setEdgesMap(CDT::Triangulation<double>& cdt) {

        auto cdtExt = CDT::extractEdgesFromTriangles(cdt.triangles);
        cout << "[TriangulationMesh] cdtExt size:" << cdtExt.size() << endl;


        for (auto& e : cdtExt) {
            auto edgePtr = std::make_shared<Edge>();
            auto edgeIndexPair = std::make_pair(e.v1(), e.v2());

            // [debug use]
            //if (edgeIndexPair.first >= 544 || edgeIndexPair.second >= 544)
            //{
            //    cout << "[TriangulationMesh] edgeIndexPair>=544 found:" << edgeIndexPair.first << " " << edgeIndexPair.second << endl;
            //}

            edgePtr->setVertexIndex(edgeIndexPair);
            edgePtr->setControlPoint(this->pointsList[edgeIndexPair.first]->coordinate, this->pointsList[edgeIndexPair.second]->coordinate);

            //update Vertex incidentEdgesList
            this->pointsList[edgeIndexPair.first]->incidentEdgesList.emplace_back(edgePtr);
            this->pointsList[edgeIndexPair.second]->incidentEdgesList.emplace_back(edgePtr);

            edgesMap[edgeIndexPair] = edgePtr;
        }

    }

    void _setTrianglesList(CDT::Triangulation<double>& cdt) {
        auto cdtTri = cdt.triangles;
        cout << "[TriangulationMesh] cdtTri size:" << cdtTri.size() << endl;

        for (auto& t : cdtTri) {
            auto triPtr = std::make_shared<CurvedTriangle>();

            for (auto& verticesIndex : t.vertices) {
                triPtr->verticesIndexList.emplace_back(verticesIndex);
            }

            for (auto& triIndex : t.vertices) {
                triPtr->incidentTrianglesIndexList.emplace_back(triIndex);
            }

            // update edges left/right triangle
            for (size_t i = 0; i < t.vertices.size(); i++) {
                size_t j = i + 1;
                if (j == t.vertices.size()) {
                    j = 0;
                }

                auto p1Index = t.vertices[i];
                auto p2Index = t.vertices[j];

                auto it = edgesMap.find({ p1Index, p2Index });
                bool isRight = false;
                if (it == edgesMap.end()) {
                    it = edgesMap.find({ p2Index, p1Index });
                    isRight = true; //reverse to right
                }

                it->second->setTrianglePtr(triPtr, isRight);
            }

            trianglesList.emplace_back(triPtr);
        }
    }

    void _setPointsIncidentVertices() {

        uint pointIndex = 0;
        for (auto& pointPtr : this->pointsList) {
            for (auto& edgePtr : pointPtr->incidentEdgesList) {

                auto otherPointIndex = edgePtr->vertexIndexPair.first;
                if (otherPointIndex == pointIndex) {
                    otherPointIndex = edgePtr->vertexIndexPair.second;
                }

                pointPtr->incidentVerticesList.emplace_back(otherPointIndex);
            }

            // sort
            std::sort(
                pointPtr->incidentVerticesList.begin(),
                pointPtr->incidentVerticesList.end(),
                [&](const uint &aIndex, const uint &bIndex) {
                    auto aPair = this->pointsList[aIndex]->coordinate;
                    auto bPair = this->pointsList[bIndex]->coordinate;

                    aPair.first -= pointPtr->coordinate.first;
                    aPair.second -= pointPtr->coordinate.second;

                    bPair.first -= pointPtr->coordinate.first;
                    bPair.second -= pointPtr->coordinate.second;

                    double t1 = atan2(aPair.second, aPair.first);
                    double t2 = atan2(bPair.second, bPair.first);

                    return t1 < t2;

                });


            pointIndex++;
        }

    }
};

struct HalfSurfaceChecker {
    std::vector<std::pair<cv::Point2d, cv::Point2d>> segmentsList;

    // check a and b are all in same side
    int check(const cv::Point2d &a, const cv::Point2d &b) {

        int flag = 1; //1:yes 
        for (auto& segmentPair : segmentsList) {
            auto c = segmentPair.first;
            auto d = segmentPair.second;

            auto vector_cd = d - c;
            auto vector_ca = a - c;
            auto vector_cb = b - c;

            auto val1 = vector_cd.cross(vector_ca);
            auto val2 = vector_cd.cross(vector_cb);


            if ((val1)*(val2)<=0.0) {
                // flag = 0;
                return 0;
            }
        }

        return flag;
    }
};

void _printPoints(const vector<vector<Point> > &segments) {
    for (size_t i = 0; i < segments.size(); i++)
    {
        cout << "---------------" << endl;
        auto& pointsVec = segments[i];
        for (size_t j = 0; j < pointsVec.size(); j++)
        {
            cout << "("<<pointsVec[j]<<")";
        }
        cout << "---------------" << endl;
        cout << endl;
    }
}

double getDistancePTP(const cv::Point& a, const cv::Point& b) {
    return cv::norm(a - b);
}

double getDistancePTP(const cv::Point2d& a, const cv::Point2d& b) {
    return cv::norm(a - b);
}

/* 
    c->line(a, b)
*/
double getDistancePTL(const cv::Point2d& a, const cv::Point2d& b, const cv::Point2d& c) {
    // norm = || v ||
    cv::Point2d v1 = c - a;
    cv::Point2d v2 = b - a;
    double v2Norm = cv::norm(v2);
    cv::Point2d v3 = (v1.ddot(v2) / (v2Norm * v2Norm)) * v2;
    cv::Point2d v4 = v1 - v3;
    return cv::norm(v4);
}

vector<Point> DouglasPeuckerRun(vector<Point> pointList, const double& epsilon = 5.0) {

    if (pointList.size() <= 2) {
        return pointList;
    }

    // Find out point with the maximum distance
    double distanceMaximum = 0.0;
    size_t index = 0;

    for (size_t i = 1; i < pointList.size() - 1; i++) {
        double dis = getDistancePTL(pointList.front(), pointList.back(), pointList[i]);
        if (dis > distanceMaximum) {
            index = i;
            distanceMaximum = dis;
        }
    }

    vector<Point> resultList;

    if (distanceMaximum > epsilon) {
        // recursive
        auto resultList1 = DouglasPeuckerRun(vector<Point>(pointList.begin(), pointList.begin() + index + 1), epsilon);
        auto resultList2 = DouglasPeuckerRun(vector<Point>(pointList.begin() + index, pointList.end()), epsilon);

        resultList1.insert(resultList1.end(), resultList2.begin() + 1, resultList2.end());
        resultList = resultList1;
    }
    else {
        resultList.emplace_back(pointList.front());
        resultList.emplace_back(pointList.back());
    }

    return resultList;
}

void drawBezier(const cv::Point2d p1, const cv::Point2d c1, const cv::Point2d p2, cv::Mat& window, cv::Vec3b colorVec)
{
    const double deltaT = 0.0001;
    for (double t = 0.0; t <= 1.0; t += deltaT)
    {
        auto point = std::pow(1 - t, 2) * p1 + std::pow(t, 2) * p2 + 2 * t * (1 - t) * c1;
        window.at<cv::Vec3b>(point.y, point.x) = colorVec;
    }
}

void drawMesh(const cv::Mat& imageInput, TriangulationMesh& triangulationMesh, cv::Mat& edgeImageResult) {

    // draw edge
    //edgeImageResult = Mat::zeros(imageInput.size(), CV_8UC3);
    edgeImageResult = imageInput.clone();
    for (auto e : triangulationMesh.edgesMap) {

        auto p1 = triangulationMesh.pointsList[e.first.first]->coordinate;
        auto p2 = triangulationMesh.pointsList[e.first.second]->coordinate;

        auto cvp1 = cv::Point(static_cast<int>(p1.first), static_cast<int>(p1.second));
        auto cvp2 = cv::Point(static_cast<int>(p2.first), static_cast<int>(p2.second));
        auto cvc1 = cv::Point(static_cast<int>(e.second->controlPoint.first), static_cast<int>(e.second->controlPoint.second));

        //cv::line(edgeImageResult, cvp1, cvp2, { 255,255,255 });
        drawBezier(cvp1, cvc1, cvp2, edgeImageResult, {0,0,255});
    }

    //draw points
    for (auto p : triangulationMesh.pointsList) {
        auto cvp = cv::Point(static_cast<int>(p->coordinate.first), static_cast<int>(p->coordinate.second));
        cv::circle(edgeImageResult, cvp, 2, { 255,0,0 });
    }

}

void TriangulationRun(const cv::Mat &imageInput, vector<vector<Point> >& segmentsInput, cv::Mat &edgeImageResult, TriangulationMesh & triangulationMeshResult) {

    std::vector<CustomPoint2DForCDT> points;
    std::vector<CustomEdgeForCDT> edges;

    // mapping
    std::map<std::pair<int,int>, size_t> mp;
    size_t num = 0;

    // 4 corner points
    if (mp.find({ 0,0 }) == mp.end()) {
        mp[{0, 0}] = num++;
        points.push_back({ {0,0} });
    }
    if (mp.find({ imageInput.size().width-1,imageInput.size().height-1 }) == mp.end()) {
        mp[{ imageInput.size().width-1, imageInput.size().height-1 }] = num++;
        points.push_back({ { imageInput.size().width-1,imageInput.size().height-1 } });
    }
    if (mp.find({ 0,imageInput.size().height-1 }) == mp.end()) {
        mp[{ 0, imageInput.size().height-1 }] = num++;
        points.push_back({ { 0,imageInput.size().height-1 } });
    }
    if (mp.find({ imageInput.size().width-1,0 }) == mp.end()) {
        mp[{ imageInput.size().width-1, 0 }] = num++;
        points.push_back({ { imageInput.size().width-1,0 } });
    }

    auto pointToPair = [](const cv::Point& point) {
        return make_pair(point.x, point.y);
    };

    for (auto& segment: segmentsInput) {
        for (auto& point : segment) {
            auto point_tmp = pointToPair(point);
            if (mp.find(point_tmp) == mp.end()) {
                mp[point_tmp] = num++;
                points.push_back({ point });
            }
        }
    }

    for (auto& segment : segmentsInput) {
        for (size_t j = 1; j < segment.size(); j++) {

            auto findIndex1 = mp[pointToPair(segment[j - 1])];
            auto findIndex2 = mp[pointToPair(segment[j])];

            edges.push_back({ {findIndex1, findIndex2} });
        }
    }

    cout << "[TriangulationRun] points size:" << points.size() << endl;
    cout << "[TriangulationRun] edges size:" << edges.size() << endl;


    // CDT Run
    CDT::Triangulation<double> cdt(
        CDT::VertexInsertionOrder::Auto,
        CDT::IntersectingConstraintEdges::Resolve,
        1.0
    );
    cdt.insertVertices(
        points.begin(),
        points.end(),
        [](const CustomPoint2DForCDT& p) { return static_cast<double>(p.data.x); },
        [](const CustomPoint2DForCDT& p) { return static_cast<double>(p.data.y); }
    );
    cdt.insertEdges(
        edges.begin(),
        edges.end(),
        [](const CustomEdgeForCDT& e) { return e.vertices.first; },
        [](const CustomEdgeForCDT& e) { return e.vertices.second; }
    );

    cdt.eraseSuperTriangle();

    TriangulationMesh triangulationMesh(cdt);

    // draw edge (no necessary)
    //edgeImageResult = Mat::zeros(imageInput.size(), CV_8UC3);
    edgeImageResult = imageInput.clone();
    for (auto e : triangulationMesh.edgesMap) {

        auto p1 = triangulationMesh.pointsList[e.first.first]->coordinate;
        auto p2 = triangulationMesh.pointsList[e.first.second]->coordinate;

        auto cvp1 = cv::Point(static_cast<int>(p1.first), static_cast<int>(p1.second));
        auto cvp2 = cv::Point(static_cast<int>(p2.first), static_cast<int>(p2.second));

        // [debug use]
        if (e.first.first >= points.size() || e.first.second >= points.size())
        {
            cv::line(edgeImageResult, cvp1, cvp2, { 255,0,0 });
        }
        else 
        {
            cv::line(edgeImageResult, cvp1, cvp2, { 0,0,255 });
        }
    }

    //draw points
    int _pcnt = 0;
    for (auto p : triangulationMesh.pointsList) {
        auto cvp = cv::Point(static_cast<int>(p->coordinate.first), static_cast<int>(p->coordinate.second));

        
        if (_pcnt >= points.size())
        {
            cv::circle(edgeImageResult, cvp, 2, { 0,0,255 });
        }
        else
        {
            cv::circle(edgeImageResult, cvp, 2, { 255,0,0 });
        }

        _pcnt++;
    }

    triangulationMeshResult = triangulationMesh;
}

void edgeDrawingRun(cv::Mat imageInput, cv::Mat &imageResult, vector<vector<Point> > &segmentsResult) {

    // convert
    cv::Mat image;
    cv::cvtColor(imageInput, image, COLOR_BGR2GRAY);

    // edge drawing
    Ptr<EdgeDrawing> ed = createEdgeDrawing();
    //ed->params.MaxDistanceBetweenTwoLines = 200.0;
    ed->params.EdgeDetectionOperator = EdgeDrawing::SOBEL;
    ed->params.GradientThresholdValue = 38;
    ed->params.AnchorThresholdValue = 8;
    ed->params.MinPathLength = 30;

    vector<Vec4f> lines;
    lines.clear();
    // Detect edges
    //you should call this before detectLines() and detectEllipses()
    ed->detectEdges(image);
    // Detect lines
    ed->detectLines(lines);

    // draw lines
    Mat edge_image_ed = Mat::zeros(image.size(), CV_8UC3);
    vector<vector<Point> > segments = ed->getSegments();

    // DouglasPeuckerRun
    for (size_t i = 0; i < segments.size(); i++)
    {
        //size_t beforDPNum = segments[i].size();
        segments[i] = DouglasPeuckerRun(segments[i]);
        //size_t afterDPNum = segments[i].size();
        //cout << "++ " << afterDPNum << " : " << beforDPNum << " ++" << endl;
    }

    // draw segments to result mat
    for (size_t i = 0; i < segments.size(); i++)
    {
        const Point* pts = &segments[i][0];
        int n = (int)segments[i].size();
        //random color
        polylines(edge_image_ed, &pts, &n, 1, false, Scalar((rand() & 255), (rand() & 255), (rand() & 255)), 1);
    }

    imageResult = edge_image_ed;
    segmentsResult = segments;
}

void rasterization(const cv::Mat& imageInput, TriangulationMesh& triangulationMesh, bool debug_flag = false, std::string imName = "[rasterization] imageResult") {

    int _cnt1 = 0;
    const int VERTICES_SIZE = 3;
    const int SCALE_VALUE = 10;
    cv::Mat imageResult = cv::Mat::zeros(imageInput.size(), CV_8UC3);

    for (auto& triPtr : triangulationMesh.trianglesList) {
        cv::Mat tmpMat = cv::Mat::zeros(imageInput.size() * SCALE_VALUE, CV_8UC3);

        cv::Point2d centerPoint;
        double minx = imageInput.size().width + 10.0;
        double miny = imageInput.size().height + 10.0;
        double maxx = 0.0;
        double maxy = 0.0;

        // draw 3 bezier on tmpMat
        for (size_t i = 0; i < VERTICES_SIZE; i++) {
            size_t j = i + 1;
            if (j == VERTICES_SIZE) {
                j = 0;
            }

            auto v1Index = triPtr->verticesIndexList[i];
            auto v2Index = triPtr->verticesIndexList[j];

            auto it = triangulationMesh.edgesMap.find({ v1Index, v2Index });
            if (it == triangulationMesh.edgesMap.end()) {
                it = triangulationMesh.edgesMap.find({ v2Index, v1Index });
            }

            auto p1Ptr = triangulationMesh.pointsList[v1Index];
            auto p2Ptr = triangulationMesh.pointsList[v2Index];

            //update center point
            //centerPoint.x += p1Ptr->coordinate.first;
            //centerPoint.y += p1Ptr->coordinate.second;

            centerPoint.x += it->second->controlPoint.first;
            centerPoint.y += it->second->controlPoint.second;

            //update boundbox
            {
                minx = std::min(minx, p1Ptr->coordinate.first);
                miny = std::min(miny, p1Ptr->coordinate.second);
                maxx = std::max(maxx, p1Ptr->coordinate.first);
                maxy = std::max(maxy, p1Ptr->coordinate.second);

                minx = std::min(minx, p2Ptr->coordinate.first);
                miny = std::min(miny, p2Ptr->coordinate.second);
                maxx = std::max(maxx, p2Ptr->coordinate.first);
                maxy = std::max(maxy, p2Ptr->coordinate.second);

                minx = std::min(minx, it->second->controlPoint.first);
                miny = std::min(miny, it->second->controlPoint.second);
                maxx = std::max(maxx, it->second->controlPoint.first);
                maxy = std::max(maxy, it->second->controlPoint.second);
            }


            auto p1cv = cv::Point2d(p1Ptr->coordinate.first, p1Ptr->coordinate.second);
            auto p2cv = cv::Point2d(p2Ptr->coordinate.first, p2Ptr->coordinate.second);
            auto c1cv = cv::Point2d(it->second->controlPoint.first, it->second->controlPoint.second);

            // to ensure drawing in same order
            if (v1Index > v2Index) {
                std::swap(p1cv, p2cv);
            }

            drawBezier(p1cv * SCALE_VALUE, c1cv * SCALE_VALUE, p2cv * SCALE_VALUE, tmpMat, {255,255,255});


        }

        centerPoint /= VERTICES_SIZE;

        //adjust bound box
        {
            minx -= 1.0;
            miny -= 1.0;
            maxx += 1.0;
            maxy += 1.0;

            minx = std::max(minx, 0.0);
            miny = std::max(miny, 0.0);
            maxx = std::min(maxx, (imageInput.size().width-1) * 1.0);
            maxy = std::min(maxy, (imageInput.size().height-1) *1.0);   
        }

        cv::floodFill(tmpMat, centerPoint * SCALE_VALUE, { 255,255,255 }, (cv::Rect*)0, { 5,5,5 }, {5,5,5});
        
        //// [debug use]
        //if (debug_flag && _cnt1 == 109)
        //{

        //    cv::circle(tmpMat, centerPoint * SCALE_VALUE, 50, { 0,0,255 }); // ??

        //    cv::Mat tmp2;
        //    cv::resize(tmpMat, tmp2, imageInput.size()*2);

        //    cv::imshow("[rasterization] tmp2", tmp2);
        //    cv::waitKey(0);

        //}

        // scan every pixels
        cv::Vec3i sumVec;
        int sumcnt = 0;

        triPtr->pixelsList.clear();
        for (uint j = miny; j <= maxy; j++) { //row
            for (uint i = minx; i <= maxx; i++) { //col
                if (tmpMat.at<cv::Vec3b>(j*SCALE_VALUE, i*SCALE_VALUE) == cv::Vec3b(255, 255, 255))
                {
                    sumVec += imageInput.at<cv::Vec3b>(j, i);
                    sumcnt++;

                    triPtr->pixelsList.emplace_back(i, j); // coordinate
                }
            }
        }

        sumVec /= sumcnt;
        triPtr->constColor = sumVec;

        for (auto &coordinatePair: triPtr->pixelsList)
        {
            auto x = coordinatePair.first;
            auto y = coordinatePair.second;

            imageResult.at<cv::Vec3b>(y, x) = triPtr->constColor;
        }


        _cnt1++;
        std::cout << "[rasterization] _cnt1: " << _cnt1 <<" "<< triangulationMesh.trianglesList.size() << std::endl;
    }


    // debug
    cv::imshow(imName, imageResult);
    cv::waitKey(0);
}

void vertexGradient(const cv::Mat& imageInput, TriangulationMesh& triangulationMesh) {

    const double deltaT = 0.001;


    for (size_t i = 0; i < triangulationMesh.pointsList.size(); i++){
        auto pointPtr = triangulationMesh.pointsList[i];
        cv::Point2d cvpi(pointPtr->coordinate.first, pointPtr->coordinate.second);
        double safety_dis = MAXDIS;


        // check null triangle (temp)
        bool checkNullTriangleFlag = false;
        for (auto& edgePtr : pointPtr->incidentEdgesList) {
            if (edgePtr->leftTriangle == nullptr || edgePtr->rightTriangle == nullptr) {
                std::cout << "[vertexGradient] skipped" << std::endl;
                checkNullTriangleFlag = true;
                break;
            }
        }
        if (checkNullTriangleFlag) continue;

        // calculate
        cv::Point2d sumGradient(0.0,0.0);


        for (auto& edgePtr : pointPtr->incidentEdgesList) {
            auto leftTriPtr = edgePtr->leftTriangle;
            auto rightTriPtr = edgePtr->rightTriangle;


            if (edgePtr->vertexIndexPair.first != i) {
                std::swap(leftTriPtr, rightTriPtr);
            }

            auto otherPointPtr = triangulationMesh.pointsList[edgePtr->vertexIndexPair.second];

            // construct CV point (for easy calculation)
            //cv::Point2d cvp1(pointPtr->coordinate.first, pointPtr->coordinate.second);
            cv::Point2d cvp2(otherPointPtr->coordinate.first, otherPointPtr->coordinate.second);
            cv::Point2d cvc1(edgePtr->controlPoint.first, edgePtr->controlPoint.second);

            safety_dis = std::min(safety_dis, getDistancePTP(cvpi, cvc1));

            for (double t = 0.0;t<=1.0; t += deltaT) {

                // bezier
                auto cvx = std::pow(1 - t, 2) * cvpi + std::pow(t, 2) * cvp2 + 2 * t * (1 - t) * cvc1;

                // deltax
                double deltax = 0.0;
                for (uint channelId = 0; channelId < 3; channelId++) {
                    double hx = imageInput.at<cv::Vec3b>(cvx.y, cvx.x)[channelId] / 256.0;
                    deltax += std::pow((hx - leftTriPtr->constColor[channelId] / 256.0), 2);
                    deltax -= std::pow((hx - rightTriPtr->constColor[channelId] / 256.0), 2);
                }
                
                deltax *= (1 - t) * (1 - t) * deltaT;

                // gradient direction
                auto dir = -2 * (1 - t) * cvpi + (2 - 4 * t) * cvc1 + 2 * t * cvp2;
                cv::Point2d dir2(dir.y, -dir.x); // rotate 270 (rotate 90 clockwise)

                sumGradient += dir2 * deltax;
            }
        }

        if (sumGradient == cv::Point2d(0.0, 0.0)) {
            std::cout << "[vertexGradient] sumGradient == 0, i:"<< i << endl;
            continue;
        }

        sumGradient /= cv::norm(sumGradient);

        // half check
        HalfSurfaceChecker halfSurfaceChecker;

        for (size_t j = 0; j < pointPtr->incidentVerticesList.size(); j++) {
            if (pointPtr->incidentVerticesList.size() == 1) break;

            size_t k = j + 1;
            if (k == pointPtr->incidentVerticesList.size()) {
                k = 0;
            }

            auto jIndex = pointPtr->incidentVerticesList[j];
            auto kIndex = pointPtr->incidentVerticesList[k];

            auto it = triangulationMesh.edgesMap.find({ jIndex, kIndex });
            if (it == triangulationMesh.edgesMap.end()) {
                it = triangulationMesh.edgesMap.find({ kIndex, jIndex });
            }

            auto jCoord = triangulationMesh.pointsList[jIndex]->coordinate;
            auto kCoord = triangulationMesh.pointsList[kIndex]->coordinate;
            auto cCoord = it->second->controlPoint;


            // jk
            halfSurfaceChecker.segmentsList.emplace_back(
                std::make_pair(
                    cv::Point2d(jCoord.first, jCoord.second),
                    cv::Point2d(kCoord.first, kCoord.second)
                )
            );

            // jc
            halfSurfaceChecker.segmentsList.emplace_back(
                std::make_pair(
                    cv::Point2d(jCoord.first, jCoord.second),
                    cv::Point2d(cCoord.first, cCoord.second)
                )
            );

            // kc
            halfSurfaceChecker.segmentsList.emplace_back(
                std::make_pair(
                    cv::Point2d(kCoord.first, kCoord.second),
                    cv::Point2d(cCoord.first, cCoord.second)
                )
            );

            //update safety dis
            safety_dis = std::min(safety_dis, getDistancePTP(cvpi, cv::Point2d(cCoord.first, cCoord.second)));
        }

        // do gradient decrease (5 time)
        double alpha_star = 0.8 * safety_dis;
        cv::Point2d cvpi_new = cvpi;
        uint ti = 0;
        for (;ti<5;ti++) {
            cvpi_new = cvpi - alpha_star * sumGradient;

            if (halfSurfaceChecker.check(cvpi, cvpi_new) == 1) {
                break;
            }

            std::cout<<"[vertexGradient] halfSurfaceChecker failed: "
                << ti<<" "
                << cvpi_new.x << " " << cvpi_new.y << " <- "
                << cvpi.x << " " << cvpi.y
                << std::endl;

            alpha_star *= 0.8;
        }

        //update
        if (ti<5)
        {
            pointPtr->coordinate = { cvpi_new.x, cvpi_new.y };
            std::cout << "[vertexGradient] update success:"
                << cvpi_new.x << " "<< cvpi_new.y<<" <- "
                << cvpi.x << " " << cvpi.y
                << std::endl;
        }


        std::cout << "[vertexGradient] i: " << i << " " << triangulationMesh.pointsList.size() << std::endl;

    }
}


double controlPointEnergy(
    const cv::Mat& imageInput,
    TriangulationMesh& triangulationMesh,
    std::shared_ptr<TriangulationMesh::CurvedTriangle> leftTriPtr,
    std::shared_ptr<TriangulationMesh::CurvedTriangle> rightTriPtr,
    bool recalFlag
)
{
    const int VERTICES_SIZE = 3;
    const int SCALE_VALUE = 10;
    const std::vector<std::shared_ptr<TriangulationMesh::CurvedTriangle>> tmpTriList{ leftTriPtr , rightTriPtr };

    if (recalFlag) {
        // re-calculate (almost same as rasterization)
        for (auto& triPtr : tmpTriList) {
            cv::Mat tmpMat = cv::Mat::zeros(imageInput.size() * SCALE_VALUE, CV_8UC3);

            cv::Point2d centerPoint;
            double minx = imageInput.size().width + 10.0;
            double miny = imageInput.size().height + 10.0;
            double maxx = 0.0;
            double maxy = 0.0;

            // draw 3 bezier on tmpMat
            for (size_t i = 0; i < VERTICES_SIZE; i++) {
                size_t j = i + 1;
                if (j == VERTICES_SIZE) {
                    j = 0;
                }

                auto v1Index = triPtr->verticesIndexList[i];
                auto v2Index = triPtr->verticesIndexList[j];

                auto it = triangulationMesh.edgesMap.find({ v1Index, v2Index });
                if (it == triangulationMesh.edgesMap.end()) {
                    it = triangulationMesh.edgesMap.find({ v2Index, v1Index });
                }

                auto p1Ptr = triangulationMesh.pointsList[v1Index];
                auto p2Ptr = triangulationMesh.pointsList[v2Index];

                //update center point
                centerPoint.x += it->second->controlPoint.first;
                centerPoint.y += it->second->controlPoint.second;

                //update boundbox
                {
                    minx = std::min(minx, p1Ptr->coordinate.first);
                    miny = std::min(miny, p1Ptr->coordinate.second);
                    maxx = std::max(maxx, p1Ptr->coordinate.first);
                    maxy = std::max(maxy, p1Ptr->coordinate.second);

                    minx = std::min(minx, p2Ptr->coordinate.first);
                    miny = std::min(miny, p2Ptr->coordinate.second);
                    maxx = std::max(maxx, p2Ptr->coordinate.first);
                    maxy = std::max(maxy, p2Ptr->coordinate.second);

                    minx = std::min(minx, it->second->controlPoint.first);
                    miny = std::min(miny, it->second->controlPoint.second);
                    maxx = std::max(maxx, it->second->controlPoint.first);
                    maxy = std::max(maxy, it->second->controlPoint.second);
                }

                auto p1cv = cv::Point2d(p1Ptr->coordinate.first, p1Ptr->coordinate.second);
                auto p2cv = cv::Point2d(p2Ptr->coordinate.first, p2Ptr->coordinate.second);
                auto c1cv = cv::Point2d(it->second->controlPoint.first, it->second->controlPoint.second);

                // to ensure drawing in same order
                if (v1Index > v2Index) {
                    std::swap(p1cv, p2cv);
                }

                drawBezier(p1cv * SCALE_VALUE, c1cv * SCALE_VALUE, p2cv * SCALE_VALUE, tmpMat, { 255,255,255 });

            }

            centerPoint /= VERTICES_SIZE;

            //adjust bound box
            {
                minx -= 1.0;
                miny -= 1.0;
                maxx += 1.0;
                maxy += 1.0;

                minx = std::max(minx, 0.0);
                miny = std::max(miny, 0.0);
                maxx = std::min(maxx, (imageInput.size().width - 1) * 1.0);
                maxy = std::min(maxy, (imageInput.size().height - 1) * 1.0);
            }

            cv::floodFill(tmpMat, centerPoint * SCALE_VALUE, { 255,255,255 }, (cv::Rect*)0, { 5,5,5 }, { 5,5,5 });

            // [debug use]
            
            //{

            //    cv::Mat tmp2;
            //    cv::resize(tmpMat, tmp2, imageInput.size());
            //    cv::circle(tmp2, centerPoint, 3, { 0,0,255 });

            //    cv::imshow("[controlPointEnergy] tmp2", tmp2);
            //    cv::waitKey(0);
            //}

            // scan every pixels and update constColor for this tri
            cv::Vec3i sumVec;
            int sumcnt = 0;

            triPtr->pixelsList.clear();
            for (uint j = miny; j <= maxy; j++) { //row
                for (uint i = minx; i <= maxx; i++) { //col
                    if (tmpMat.at<cv::Vec3b>(j * SCALE_VALUE, i * SCALE_VALUE) == cv::Vec3b(255, 255, 255))
                    {
                        sumVec += imageInput.at<cv::Vec3b>(j, i);
                        sumcnt++;

                        triPtr->pixelsList.emplace_back(i, j); // coordinate
                    }
                }
            }

            sumVec /= sumcnt;
            triPtr->constColor = sumVec;
        }
    }



    double energySum = 0.0;

    // energy for left and right triangle
    for (auto& triPtr : tmpTriList) {
        for (auto& coordinatePair : triPtr->pixelsList)
        {
            auto x = coordinatePair.first;
            auto y = coordinatePair.second;

            for (uint channelId = 0; channelId < 3; channelId++) {

                double hx = imageInput.at<cv::Vec3b>(y, x)[channelId] / 256.0;
                energySum += std::pow((hx - leftTriPtr->constColor[channelId] / 256.0), 2);
            }

        }
    }

    return energySum;

}

void controlPointGradient(const cv::Mat& imageInput, TriangulationMesh& triangulationMesh) {
    const double deltaT = 0.001;
    uint ii = 0;
    uint success_cnt = 0;
    for (auto& edgeIt : triangulationMesh.edgesMap) {
        auto edgePtr = edgeIt.second;

        if (edgePtr->leftTriangle == nullptr || edgePtr->rightTriangle == nullptr) {
            std::cout << "[controlPointGradient] nullptr skipped, ii:"<< ii++ << std::endl;
            continue;
        }

        double safety_dis = MAXDIS;

        auto p1Index = edgePtr->vertexIndexPair.first;
        auto p2Index = edgePtr->vertexIndexPair.second;
        auto p1Ptr = triangulationMesh.pointsList[p1Index];
        auto p2Ptr = triangulationMesh.pointsList[p2Index];
        auto leftTriPtr = edgePtr->leftTriangle;
        auto rightTriPtr = edgePtr->rightTriangle;

        cv::Point2d cvp1(p1Ptr->coordinate.first, p1Ptr->coordinate.second);
        cv::Point2d cvp2(p2Ptr->coordinate.first, p2Ptr->coordinate.second);
        cv::Point2d cvc1 = edgePtr->getControlPointCV();

        //update safety_dis
        //safety_dis = std::min(safety_dis, getDistancePTP(cvp1, cvc1));
        //safety_dis = std::min(safety_dis, getDistancePTP(cvp2, cvc1));

        cv::Point2d sumGradient(0.0, 0.0);

        for (double t = 0.0; t <= 1.0; t += deltaT) {
            auto cvx = std::pow(1 - t, 2) * cvp1 + std::pow(t, 2) * cvp2 + 2 * t * (1 - t) * cvc1;

            // deltax
            double deltax = 0.0;
            for (uint channelId = 0; channelId < 3; channelId++) {
                double hx = imageInput.at<cv::Vec3b>(cvx.y, cvx.x)[channelId] / 256.0;
                deltax += std::pow((hx - leftTriPtr->constColor[channelId] / 256.0), 2);
                deltax -= std::pow((hx - rightTriPtr->constColor[channelId] / 256.0), 2);
            }

            deltax *= 2*t* (1 - t) * deltaT;
            // gradient direction
            auto dir = -2 * (1 - t) * cvp1 + (2 - 4 * t) * cvc1 + 2 * t * cvp2;
            cv::Point2d dir2(dir.y, -dir.x); // rotate 270 (rotate 90 clockwise)

            sumGradient += dir2 * deltax;
        }

        if (sumGradient == cv::Point2d(0.0, 0.0)) {
            std::cout << "[controlPointGradient] sumGradient == 0, ii:" << ii++ << endl;
            continue;
        }

        sumGradient /= cv::norm(sumGradient);

        // half check
        HalfSurfaceChecker halfSurfaceChecker;

        auto addConstraint = [&](std::shared_ptr<TriangulationMesh::CurvedTriangle> triPtr) {

            // get k index
            uint kIndex = triangulationMesh.pointsList.size()+100;
            for (auto index: triPtr->verticesIndexList) {
                if (index != p1Index && index != p2Index) {
                    kIndex = index;
                    break;
                }
            }

            auto kPtr = triangulationMesh.pointsList[kIndex];
            cv::Point2d cvpk(kPtr->coordinate.first, kPtr->coordinate.second);

            halfSurfaceChecker.segmentsList.emplace_back(
                std::make_pair(cvp1,cvpk)
            );

            halfSurfaceChecker.segmentsList.emplace_back(
                std::make_pair(cvp2, cvpk)
            );

            // find edge ik
            auto it = triangulationMesh.edgesMap.find({ p1Index,kIndex });
            if (it == triangulationMesh.edgesMap.end()) {
                it = triangulationMesh.edgesMap.find({ kIndex,p1Index });
            }

            cv::Point2d cvc2(it->second->controlPoint.first, it->second->controlPoint.second);
            halfSurfaceChecker.segmentsList.emplace_back(
                std::make_pair(
                    cvp1,
                    cvc2
                )
            );

            // find edge jk
            it = triangulationMesh.edgesMap.find({ p2Index,kIndex });
            if (it == triangulationMesh.edgesMap.end()) {
                it = triangulationMesh.edgesMap.find({ kIndex,p2Index });
            }

            cv::Point2d cvc3(it->second->controlPoint.first, it->second->controlPoint.second);
            halfSurfaceChecker.segmentsList.emplace_back(
                std::make_pair(
                    cvp2,
                    cvc3
                )
            );

            //update safety_dis
            //safety_dis = std::min(safety_dis, getDistancePTP(cvpk, cvc1));
            //safety_dis = std::min(safety_dis, getDistancePTP(cvc2, cvc1));
            //safety_dis = std::min(safety_dis, getDistancePTP(cvc3, cvc1));
        };

        addConstraint(leftTriPtr);
        addConstraint(rightTriPtr);

        // get max dis can move for half surface
        cv::Point2d cvc1_tmp = cvc1 - sumGradient * safety_dis;
        while (halfSurfaceChecker.check(cvc1, cvc1_tmp) != 1 && (safety_dis>0.001)) {
            safety_dis /= 2.0;
            cvc1_tmp = cvc1 - sumGradient * safety_dis;
        }

        if (safety_dis <= 0.001) {
            std::cout << "[controlPointGradient] safety_dis <= 0.001, update skipped, ii:" << ii++ << endl;
            continue;
        }

        // do gradient decrease (5 time)
        double energyOrigin = controlPointEnergy(imageInput, triangulationMesh, leftTriPtr, rightTriPtr, false);
        const double CCC = 0.2;
        const uint TRY_TIMES = 5;
        const uint energyZeroCnt_UPBOUND = 2;

        double alpha_star = CCC * safety_dis;
        cv::Point2d cvc1_new = cvc1;
        uint ti = 0;
        uint energyZeroCnt = 0;
        for (; ti < TRY_TIMES; ti++) {
            cvc1_new = cvc1 - alpha_star * sumGradient;

            if (halfSurfaceChecker.check(cvc1, cvc1_new) != 1) {

                std::cout << "[controlPointGradient] halfSurfaceChecker failed: "
                    << ti << " "
                    << cvc1_new.x << " " << cvc1_new.y << " <- "
                    << cvc1.x << " " << cvc1.y
                    << std::endl;

                alpha_star *= CCC;
                continue;
            }

            // check energy decrease
            // set new cvc1_new to mesh
            edgePtr->setControlPoint(cvc1_new);

            double energyNew = controlPointEnergy(imageInput, triangulationMesh, leftTriPtr, rightTriPtr, true);
            double energyDelta = energyNew - energyOrigin;

            std::cout << "[controlPointGradient] energy delta:" << (energyDelta)<<"="<<(energyNew)<<"-"<<(energyOrigin) << std::endl;


            if (energyDelta < 0.0) {
                std::cout << "[controlPointGradient] update success:"
                    << ti << " "
                    << cvc1_new.x << " " << cvc1_new.y << " <- "
                    << cvc1.x << " " << cvc1.y
                    << std::endl;

                // accept this set and break;
                success_cnt++;
                    break;
            }
            else if (energyDelta == 0.0){
                energyZeroCnt++;
                if (energyZeroCnt == energyZeroCnt_UPBOUND) {
                    break;
                }
            }

            alpha_star *= CCC;
        }

        if (ti == TRY_TIMES || energyZeroCnt == energyZeroCnt_UPBOUND) {
            // undo set cvc1_new
            edgePtr->setControlPoint(cvc1);

            controlPointEnergy(imageInput, triangulationMesh, leftTriPtr, rightTriPtr, true);

            std::cout << "[controlPointGradient] update failed:"
                << ti << " "
                << cvc1_new.x << " " << cvc1_new.y << " <- "
                << cvc1.x << " " << cvc1.y
                << std::endl;
        }

        std::cout << "[controlPointGradient] Finish ii: " << ii++ << " " << triangulationMesh.edgesMap.size() << std::endl;
    }

    // TODO:static...
    std::cout << "[controlPointGradient] All Finished, success_cnt: " << success_cnt << std::endl;

}


int main2()
{
    //String imgPath = "D:\\图\\QQ图片20230323001527.jpg"; //zhong
    //String imgPath = "D:\\图\\QQ图片20230321000528.jpg"; //erciyuan
    String imgPath = "D:\\图\\Snipaste_2023-03-29_23-14-29.jpg"; //siyecao
    

    Mat image;
    image = imread(imgPath, IMREAD_COLOR); // Read the file
    if (image.empty()) // Check for invalid input
    {
        cout << "[main2] Could not open or find the image" << std::endl;
        return -1;
    }

    // process
    {
        cout << "[main2] COLS:" << image.cols << endl;
        cout << "[main2] ROWS:" << image.rows << endl;
        cout << "[main2] CHANNELS:" << image.channels() << endl;

        // 1. edge drawing
        cv::Mat result_EdgeDrawing_mat;
        vector<vector<Point> > result_EdgeDrawing_segments;
        edgeDrawingRun(image, result_EdgeDrawing_mat, result_EdgeDrawing_segments);

        cv::imshow("[main2] EdgeDrawing detected edges", result_EdgeDrawing_mat);
        cv::waitKey(0);

        // 2. triangulation
        cv::Mat result_Triangulation_mat;
        TriangulationMesh result_triangulationMesh;
        TriangulationRun(image, result_EdgeDrawing_segments, result_Triangulation_mat, result_triangulationMesh);
        cv::imshow("[main2] TriangulationImage", result_Triangulation_mat);
        cv::waitKey(0);

        // 3. rasterization
        rasterization(image, result_triangulationMesh);

        // 4. vetices update
        //vertexGradient(image, result_triangulationMesh);
        //cv::Mat result_Triangulation_mat2;
        //drawMesh(image, result_triangulationMesh, result_Triangulation_mat2);
        //cv::imshow("[main2] vertexGradient Image", result_Triangulation_mat2);
        //cv::waitKey(0);

        // 5. control point update

        const uint it_i_BOUND = 10;
        for (uint it_i = 0; it_i< it_i_BOUND; it_i++)
        {
            controlPointGradient(image, result_triangulationMesh);
            cv::Mat result_Triangulation_mat3;
            drawMesh(image, result_triangulationMesh, result_Triangulation_mat3);

            cout << "[main2] UPDATE it_i:" << it_i << endl;

            std::string imName = "[main2] controlPointGradient Image";
            imName += std::to_string(it_i);

            string imName2 = "[rasterization] imageResult";
            imName2 += std::to_string(it_i);
                 
            cv::imshow(imName, result_Triangulation_mat3);
            //cv::waitKey(0);

            rasterization(image, result_triangulationMesh,false, imName2);

        }
    }


    return 0;
}


int _test() {
    cv::Point c(2, 2), a(4, -3), b(6, 2);
    cout << getDistancePTL(a, b, c) << endl;

    return 0;
}

int _test2() {
    std::vector<CustomPoint2DForCDT> points;
    std::vector<CustomEdgeForCDT> edges;

    //0~5
    points.push_back({ { 0,0 } });
    points.push_back({ { 10,0 } });
    points.push_back({ { 10,10 } });
    points.push_back({ { 0,10 } });
    points.push_back({ { 2,5 } });
    points.push_back({ { 8,5 } });

    //6~7
    points.push_back({ { 5,2 } });
    points.push_back({ { 5,8 } });


    //edges.push_back({ {4,5} });
    //edges.push_back({ {1,3} });
    //edges.push_back({ {0,2} });
    //edges.push_back({ {0,1} });
    //edges.push_back({ {1,2} });
    //edges.push_back({ {2,3} });
    //edges.push_back({ {3,0} });
    edges.push_back({ {4,5} });
    edges.push_back({ {6,7} });



    CDT::Triangulation<double> cdt(
        CDT::VertexInsertionOrder::Auto,
        CDT::IntersectingConstraintEdges::Resolve,
        1.0
    );
    cdt.insertVertices(
        points.begin(),
        points.end(),
        [](const CustomPoint2DForCDT& p) { return static_cast<double>(p.data.x); },
        [](const CustomPoint2DForCDT& p) { return static_cast<double>(p.data.y); }
    );
    cdt.insertEdges(
        edges.begin(),
        edges.end(),
        [](const CustomEdgeForCDT& e) { return e.vertices.first; },
        [](const CustomEdgeForCDT& e) { return e.vertices.second; }
    );

    cdt.eraseSuperTriangle();

    TriangulationMesh triangulationMesh(cdt);



    return 0;
}

int main(int argc, char** argv)
{
    //return _test2();
    return main2();
    //return _test();
    return 0;
}