/*
    image triangulation for const color
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

const double MAXDIS = 99999.0;
const uint DESIRE_VERTICES_NUM = 130;
vector<uint> success_cnt_vertex, success_cnt_controlPoint;
vector<double> energy_vec, rmse_vec;

struct CustomPoint2DForCDT
{
    cv::Point data;
};

struct CustomEdgeForCDT
{
    std::pair<std::size_t, std::size_t> vertices;
};

void drawBezier(const cv::Point2d p1, const cv::Point2d c1, const cv::Point2d p2, cv::Mat& window, cv::Vec3b colorVec)
{
    const double deltaT = 0.0001;
    for (double t = 0.0; t <= 1.0; t += deltaT)
    {
        auto point = std::pow(1 - t, 2) * p1 + std::pow(t, 2) * p2 + 2 * t * (1 - t) * c1;
        window.at<cv::Vec3b>(point.y, point.x) = colorVec;
    }
}

struct TriangulationMesh {
    struct CurvedTriangle {
        std::vector<uint> verticesIndexList;
        //std::vector<uint> incidentTrianglesIndexList;
        std::vector<std::pair<uint, uint>> pixelsList;

        // TODO: interpolation parameters...
        // now only const

        bool constFlag = false;
        cv::Vec3i constColor;
        double xxSum = 0.0, yySum = 0.0, xySum = 0.0, xSum = 0.0, ySum = 0.0;
        double aaa[3], bbb[3], ccc[3];
        double energy = 0.0;

        cv::Vec3b getInterpolationResult(cv::Point2d p) {
            return constColor;
        }

        // call this after pixelsList was updated
        void _updateSum() {
            xxSum = 0.0;
            yySum = 0.0;
            xySum = 0.0;
            xSum = 0.0;
            ySum = 0.0;

            for (uint channelId = 0; channelId < 3; channelId++) {
                aaa[channelId] = bbb[channelId] = ccc[channelId] = 0.0;
            }

            for (auto& coorPair : pixelsList) {
                double x = static_cast<double>(coorPair.first);
                double y = static_cast<double>(coorPair.second);

                xxSum += x * x;
                yySum += y * y;
                xySum += x * y;
                xSum += x;
                ySum += y;
            }
        }

        // call this after pixelsList was updated
        void solve(const cv::Mat& imageInput) {

            // to const
            constFlag = true;

            // update constColor...

            cv::Vec3i sumVec;
            int sumCnt = pixelsList.size();
            for (auto& coordinatePair : pixelsList)
            {
                auto x = coordinatePair.first;
                auto y = coordinatePair.second;

                sumVec += imageInput.at<cv::Vec3b>(y, x);
            }

            if (sumCnt != 0) {
                sumVec /= sumCnt;
            }

            constColor = sumVec;
            return;
            
        }

        void updatePixels(const cv::Mat& imageInput, TriangulationMesh& triangulationMesh) {
            const int VERTICES_SIZE = 3;
            const int SCALE_VALUE = 10;

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

                auto v1Index = this->verticesIndexList[i];
                auto v2Index = this->verticesIndexList[j];

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

            //// [debug use]
            //if (debug_flag && _cnt1 == 109)
            //{
            //    cv::circle(tmpMat, centerPoint * SCALE_VALUE, 50, { 0,0,255 }); // ??
            //    cv::Mat tmp2;
            //    cv::resize(tmpMat, tmp2, imageInput.size()*2);
            //    cv::imshow("[rasterization] tmp2", tmp2);
            //    wt(0);
            //}

            // scan every pixels
            this->pixelsList.clear();
            for (uint j = miny; j <= maxy; j++) { //row
                for (uint i = minx; i <= maxx; i++) { //col
                    if (tmpMat.at<cv::Vec3b>(j * SCALE_VALUE, i * SCALE_VALUE) == cv::Vec3b(255, 255, 255))
                    {
                        //sumVec += imageInput.at<cv::Vec3b>(j, i);
                        //sumcnt++;

                        this->pixelsList.emplace_back(i, j); // coordinate
                    }
                }
            }
        }
    };



    struct Edge {
        std::pair<uint, uint> vertexIndexPair;
        std::pair<double, double> controlPoint;
        //std::vector<std::pair<uint, uint>> bezierPointsList;
        std::shared_ptr<CurvedTriangle> leftTriangle;
        std::shared_ptr<CurvedTriangle> rightTriangle;

        void setVertexIndex(const std::pair<uint, uint>& p) {
            vertexIndexPair = p;
        }

        void setControlPoint(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
            controlPoint = { (p1.first + p2.first) / 2.0, (p1.second + p2.second) / 2.0 };
        }

        void setControlPoint(cv::Point2d p1, cv::Point2d p2) {
            auto p3 = (p1 + p2) / 2.0;
            controlPoint.first = p3.x;
            controlPoint.second = p3.y;
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

        void setCoordinate(cv::Point2d p) {
            coordinate = { p.x, p.y };
        }

        cv::Point2d getCoordinate() {
            return cv::Point2d(coordinate.first, coordinate.second);
        }
    };

    std::vector<std::shared_ptr<Vertex>> pointsList;
    std::map<std::pair<uint, uint>, std::shared_ptr<Edge>> edgesMap;
    std::vector<std::shared_ptr<CurvedTriangle>> trianglesList;

    TriangulationMesh() {}

    TriangulationMesh(CDT::Triangulation<double>& cdt) {

        _setPointsList(cdt);
        _setEdgesMap(cdt);
        _setTrianglesList(cdt);
        setPointsIncidentVertices();
    }

    void _setPointsList(CDT::Triangulation<double>& cdt) {

        auto cdtVer = cdt.vertices;
        std::cout << "[TriangulationMesh] cdtVer size:" << cdtVer.size() << std::endl;

        for (auto& v2d : cdtVer) {

            auto vertexPtr = std::make_shared<Vertex>();
            vertexPtr->setCoordinate(v2d.x, v2d.y);

            this->pointsList.emplace_back(vertexPtr);
        }
    }

    void _setEdgesMap(CDT::Triangulation<double>& cdt) {

        auto cdtExt = CDT::extractEdgesFromTriangles(cdt.triangles);
        std::cout << "[TriangulationMesh] cdtExt size:" << cdtExt.size() << std::endl;


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
        std::cout << "[TriangulationMesh] cdtTri size:" << cdtTri.size() << std::endl;

        for (auto& t : cdtTri) {
            auto triPtr = std::make_shared<CurvedTriangle>();

            for (auto& verticesIndex : t.vertices) {
                triPtr->verticesIndexList.emplace_back(verticesIndex);
            }

            //for (auto& triIndex : t.vertices) {
            //    triPtr->incidentTrianglesIndexList.emplace_back(triIndex);
            //}

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

    void setPointsIncidentVertices() {

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
                [&](const uint& aIndex, const uint& bIndex) {
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
    int check(const cv::Point2d& a, const cv::Point2d& b) {

        int flag = 1; //1:yes 
        for (auto& segmentPair : segmentsList) {
            auto c = segmentPair.first;
            auto d = segmentPair.second;

            auto vector_cd = d - c;
            auto vector_ca = a - c;
            auto vector_cb = b - c;

            auto val1 = vector_cd.cross(vector_ca);
            auto val2 = vector_cd.cross(vector_cb);
            auto val3 = val1 * val2;

            if (val3 <= 0.0 || fabs(val3) <= 1e-12) {
                // flag = 0;
                return 0;
            }


        }

        return flag;
    }
};

void _printPoints(const vector<vector<Point> >& segments) {
    for (size_t i = 0; i < segments.size(); i++)
    {
        std::cout << "---------------" << std::endl;
        auto& pointsVec = segments[i];
        for (size_t j = 0; j < pointsVec.size(); j++)
        {
            std::cout << "(" << pointsVec[j] << ")";
        }
        std::cout << "---------------" << std::endl;
        std::cout << std::endl;
    }
}

void wt(int key) {
    if (key == 1)
        cv::waitKey(0);

    return;
}

void imShowAndimWrite(const cv::Mat& image, std::string imName) {
    //cv::imshow(imName, image);
    cv::imwrite(imName, image);
    wt(0);
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

double getCos(const cv::Point2d& p1, const cv::Point2d& p2, const cv::Point2d& p3) {
    auto v12 = p2 - p1;
    auto v13 = p3 - p1;
    double nv12 = cv::norm(v12);
    double nv13 = cv::norm(v13);

    return (v12.ddot(v13)) / (nv12 * nv13);
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
        drawBezier(cvp1, cvc1, cvp2, edgeImageResult, { 0,0,255 });
    }

    //draw points
    for (auto p : triangulationMesh.pointsList) {
        auto cvp = cv::Point(static_cast<int>(p->coordinate.first), static_cast<int>(p->coordinate.second));
        cv::circle(edgeImageResult, cvp, 2, { 255,0,0 });
    }

}

// call this after first rasterization
void addVertices2(const cv::Mat& imageInput, TriangulationMesh& triangulationMeshInput) {

    std::cout << "[addVertices2] addVertices2 begin" << std::endl;


    if (triangulationMeshInput.trianglesList.size() <= 3) {
        std::cout << "[addVertices2] triangulationMeshInput.trianglesList.size() <= 3" << std::endl;
        return;
    }

    int verticesNum = triangulationMeshInput.pointsList.size();

    std::cout << "[addVertices2] verticesNum (begin): " << verticesNum << std::endl;

    auto checkVaildTri = [&](std::shared_ptr<TriangulationMesh::CurvedTriangle> triPtr) {
        const int THESHOLD_PIXEL = 50;
        //const double THESHOLD_COS = 0.98480775301220805936674302458952; //10 degree (too small maybe)
        const double THESHOLD_COS = 0.86602540378443864676372317075294; //30 degree (too small maybe) 0.86602540378443864676372317075294

        // too small
        if (triPtr->pixelsList.size() < 200) {
            //std::cout << "[addVertices2] maxCosVal > THESHOLD_COS: triPtr->pixelsList.size() < 50: " << triPtr->pixelsList.size() << std::endl;
            return false;
        }

        auto p1Id = triPtr->verticesIndexList[0];
        auto p2Id = triPtr->verticesIndexList[1];
        auto p3Id = triPtr->verticesIndexList[2];
        auto p1Ptr = triangulationMeshInput.pointsList[p1Id];
        auto p2Ptr = triangulationMeshInput.pointsList[p2Id];
        auto p3Ptr = triangulationMeshInput.pointsList[p3Id];
        auto p1 = p1Ptr->getCoordinate();
        auto p2 = p2Ptr->getCoordinate();
        auto p3 = p3Ptr->getCoordinate();

        if (getDistancePTP(p1, p2) < 5.0 || getDistancePTP(p2, p3) < 5.0 || getDistancePTP(p1, p3) < 5.0) {
            //std::cout << "[addVertices2] getDistancePTP<5.0" << std::endl;
            return false;
        }


        // get smallest angle (biggest CosVal)
        double maxCosVal = 0.0; // 1:small 0:big angle

        maxCosVal = std::max(maxCosVal, getCos(p1, p2, p3));
        maxCosVal = std::max(maxCosVal, getCos(p2, p1, p3));
        maxCosVal = std::max(maxCosVal, getCos(p3, p1, p2));

        if (maxCosVal > THESHOLD_COS) {
            //std::cout << "[addVertices2] maxCosVal > THESHOLD_COS: " << maxCosVal << std::endl;
            return false;
        }

        return true;
    };

    while (verticesNum < DESIRE_VERTICES_NUM) {

        int iMax = -1;
        double energyMax = 0.0;
        for (uint i = 0; i < triangulationMeshInput.trianglesList.size(); i++) {
            auto itriPtr = triangulationMeshInput.trianglesList[i];

            if (!checkVaildTri(itriPtr)) {
                continue;
            }

            // max
            if (energyMax < itriPtr->energy) {
                iMax = i;
                energyMax = itriPtr->energy;
            }
        }

        if (iMax == -1) {
            std::cout << "[addVertices2] iMax == -1" << std::endl;
            return;
        }

        auto triPtr = triangulationMeshInput.trianglesList[iMax];
        auto p1Id = triPtr->verticesIndexList[0];
        auto p2Id = triPtr->verticesIndexList[1];
        auto p3Id = triPtr->verticesIndexList[2];
        auto p1Ptr = triangulationMeshInput.pointsList[p1Id];
        auto p2Ptr = triangulationMeshInput.pointsList[p2Id];
        auto p3Ptr = triangulationMeshInput.pointsList[p3Id];
        auto p1 = p1Ptr->getCoordinate();
        auto p2 = p2Ptr->getCoordinate();
        auto p3 = p3Ptr->getCoordinate();


        auto midp = (p1 + p2 + p3) / 3;
        auto midpId = verticesNum;

        // 1 new vertex
        auto midpPtr = std::make_shared<TriangulationMesh::Vertex>();
        midpPtr->setCoordinate(midp);
        midpPtr->incidentVerticesList.emplace_back(p1Id);
        midpPtr->incidentVerticesList.emplace_back(p2Id);
        midpPtr->incidentVerticesList.emplace_back(p3Id);
        triangulationMeshInput.pointsList.emplace_back(midpPtr);

        // 3 new edge (and update vertex)
        // midp p1
        auto midp_p1_EdgePtr = std::make_shared<TriangulationMesh::Edge>();
        midp_p1_EdgePtr->setVertexIndex({ midpId, p1Id });
        midp_p1_EdgePtr->setControlPoint(midp, p1);
        triangulationMeshInput.edgesMap[{midpId, p1Id}] = midp_p1_EdgePtr;
        midpPtr->incidentEdgesList.emplace_back(midp_p1_EdgePtr);
        p1Ptr->incidentEdgesList.emplace_back(midp_p1_EdgePtr);

        // midp p2
        auto midp_p2_EdgePtr = std::make_shared<TriangulationMesh::Edge>();
        midp_p2_EdgePtr->setVertexIndex({ midpId, p2Id });
        midp_p2_EdgePtr->setControlPoint(midp, p2);
        triangulationMeshInput.edgesMap[{midpId, p2Id}] = midp_p2_EdgePtr;
        midpPtr->incidentEdgesList.emplace_back(midp_p2_EdgePtr);
        p2Ptr->incidentEdgesList.emplace_back(midp_p2_EdgePtr);

        // midp p3
        auto midp_p3_EdgePtr = std::make_shared<TriangulationMesh::Edge>();
        midp_p3_EdgePtr->setVertexIndex({ midpId, p3Id });
        midp_p3_EdgePtr->setControlPoint(midp, p3);
        triangulationMeshInput.edgesMap[{midpId, p3Id}] = midp_p3_EdgePtr;
        midpPtr->incidentEdgesList.emplace_back(midp_p3_EdgePtr);
        p3Ptr->incidentEdgesList.emplace_back(midp_p3_EdgePtr);

        // 3 new triangle (and update edge)
        // midp p1 p2
        auto nTriPtr1 = std::make_shared<TriangulationMesh::CurvedTriangle>();
        nTriPtr1->verticesIndexList.emplace_back(midpId);
        nTriPtr1->verticesIndexList.emplace_back(p1Id);
        nTriPtr1->verticesIndexList.emplace_back(p2Id);
        for (uint ii = 0; ii < nTriPtr1->verticesIndexList.size(); ii++) {
            uint jj = ii + 1;
            if (jj == nTriPtr1->verticesIndexList.size()) {
                jj = 0;
            }

            auto iiId = nTriPtr1->verticesIndexList[ii];
            auto jjId = nTriPtr1->verticesIndexList[jj];

            auto it = triangulationMeshInput.edgesMap.find({ iiId, jjId });
            bool isRight = false;
            if (it == triangulationMeshInput.edgesMap.end()) {
                it = triangulationMeshInput.edgesMap.find({ jjId, iiId });
                isRight = true;
            }

            it->second->setTrianglePtr(nTriPtr1, isRight);
        }

        // midp p2 p3
        auto nTriPtr2 = std::make_shared<TriangulationMesh::CurvedTriangle>();
        nTriPtr2->verticesIndexList.emplace_back(midpId);
        nTriPtr2->verticesIndexList.emplace_back(p2Id);
        nTriPtr2->verticesIndexList.emplace_back(p3Id);
        for (uint ii = 0; ii < nTriPtr2->verticesIndexList.size(); ii++) {
            uint jj = ii + 1;
            if (jj == nTriPtr2->verticesIndexList.size()) {
                jj = 0;
            }

            auto iiId = nTriPtr2->verticesIndexList[ii];
            auto jjId = nTriPtr2->verticesIndexList[jj];

            auto it = triangulationMeshInput.edgesMap.find({ iiId, jjId });
            bool isRight = false;
            if (it == triangulationMeshInput.edgesMap.end()) {
                it = triangulationMeshInput.edgesMap.find({ jjId, iiId });
                isRight = true;
            }

            it->second->setTrianglePtr(nTriPtr2, isRight);
        }

        // midp p3 p1
        auto nTriPtr3 = std::make_shared<TriangulationMesh::CurvedTriangle>();
        nTriPtr3->verticesIndexList.emplace_back(midpId);
        nTriPtr3->verticesIndexList.emplace_back(p3Id);
        nTriPtr3->verticesIndexList.emplace_back(p1Id);
        for (uint ii = 0; ii < nTriPtr3->verticesIndexList.size(); ii++) {
            uint jj = ii + 1;
            if (jj == nTriPtr3->verticesIndexList.size()) {
                jj = 0;
            }

            auto iiId = nTriPtr3->verticesIndexList[ii];
            auto jjId = nTriPtr3->verticesIndexList[jj];

            auto it = triangulationMeshInput.edgesMap.find({ iiId, jjId });
            bool isRight = false;
            if (it == triangulationMeshInput.edgesMap.end()) {
                it = triangulationMeshInput.edgesMap.find({ jjId, iiId });
                isRight = true;
            }

            it->second->setTrianglePtr(nTriPtr3, isRight);
        }


        // 1 remove old triangle (and update triangle)
        triangulationMeshInput.trianglesList.erase(triangulationMeshInput.trianglesList.begin() + iMax);
        triangulationMeshInput.trianglesList.emplace_back(nTriPtr1);
        triangulationMeshInput.trianglesList.emplace_back(nTriPtr2);
        triangulationMeshInput.trianglesList.emplace_back(nTriPtr3);

        // update nTriPtr1 pixelsList & energy

        std::vector<std::shared_ptr<TriangulationMesh::CurvedTriangle>> tmpTriList{ nTriPtr1,nTriPtr2,nTriPtr3 };

        for (auto& nTriPtr : tmpTriList) {
            nTriPtr->updatePixels(imageInput, triangulationMeshInput);
            nTriPtr->solve(imageInput);

            nTriPtr->energy = 0.0;
            for (auto& coordinatePair : nTriPtr->pixelsList) {
                auto x = (coordinatePair.first);
                auto y = (coordinatePair.second);

                auto interpolationResult = nTriPtr->getInterpolationResult(cv::Point2d(x, y));
                auto originalColor = imageInput.at<cv::Vec3b>(y, x);
                for (uint channelId = 0; channelId < 3; channelId++) {
                    double hx = originalColor[channelId] / 256.0;
                    double px = interpolationResult[channelId] / 256.0;

                    double e = std::pow((hx - px), 2);
                    nTriPtr->energy += e;
                }
            }
        }

        std::cout << "[addVertices2] add success: "
            << verticesNum << " "
            << iMax << " "
            << energyMax << " "
            << p1 << " "
            << p2 << " "
            << p3 << " "
            << midp << " "
            << std::endl;

        verticesNum++;
    }

    std::cout << "[addVertices2] add finished: " << verticesNum << std::endl;


}


void TriangulationRun(const cv::Mat& imageInput, vector<vector<Point> >& segmentsInput, cv::Mat& edgeImageResult, TriangulationMesh& triangulationMeshResult) {

    std::vector<CustomPoint2DForCDT> points;
    std::vector<CustomEdgeForCDT> edges;

    // mapping
    std::map<std::pair<int, int>, size_t> mp;
    size_t num = 0;

    // 4 corner points
    if (mp.find({ 0,0 }) == mp.end()) {
        mp[{0, 0}] = num++;
        points.push_back({ {0,0} });
    }
    if (mp.find({ imageInput.size().width - 1,imageInput.size().height - 1 }) == mp.end()) {
        mp[{ imageInput.size().width - 1, imageInput.size().height - 1 }] = num++;
        points.push_back({ { imageInput.size().width - 1,imageInput.size().height - 1 } });
    }
    if (mp.find({ 0,imageInput.size().height - 1 }) == mp.end()) {
        mp[{ 0, imageInput.size().height - 1 }] = num++;
        points.push_back({ { 0,imageInput.size().height - 1 } });
    }
    if (mp.find({ imageInput.size().width - 1,0 }) == mp.end()) {
        mp[{ imageInput.size().width - 1, 0 }] = num++;
        points.push_back({ { imageInput.size().width - 1,0 } });
    }

    auto pointToPair = [](const cv::Point& point) {
        return make_pair(point.x, point.y);
    };

    for (auto& segment : segmentsInput) {
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

    std::cout << "[TriangulationRun] points size:" << points.size() << std::endl;
    std::cout << "[TriangulationRun] edges size:" << edges.size() << std::endl;


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

void addVertices1(vector<vector<cv::Point> >& segmentsInput) {
    if (segmentsInput.size() <= 3) {
        std::cout << "[addVertices1] segmentsInput.size()<=3 " << std::endl;
        return;
    }

    double lenMax = 0.0;
    double lenSum = 0.0;
    double lenAvg = 0.0;
    int verticesNum = 0;
    int lenCnt = 0;
    uint iMax = 0, jMax = 0;
    cv::Point pMax1, pMax2;

    // get info
    for (size_t i = 0; i < segmentsInput.size(); i++)
    {
        auto& seg = segmentsInput[i];
        verticesNum += seg.size();

        for (size_t j = 0; j < seg.size() - 1; j++) {
            size_t k = j + 1;

            cv::Point p1 = seg[j], p2 = seg[k];
            double dis = getDistancePTP(p1, p2);
            lenSum += dis;

            if (lenMax < dis) {
                lenMax = dis;
                iMax = i;
                jMax = j;
                pMax1 = p1;
                pMax2 = p2;
            }

            lenCnt++;
        }
    }

    //lenAvg = lenSum / lenCnt;
    lenAvg = lenMax / 4.0;

    //int originLenCnt = lenCnt;
    std::cout << "[addVertices1] origin verticesNum: " << verticesNum << std::endl;

    // re-add vertices

    while (verticesNum < DESIRE_VERTICES_NUM) {

        //find max dis seg
        lenMax = 0.0;
        iMax = jMax = 0; //?
        for (size_t i = 0; i < segmentsInput.size(); i++)
        {
            auto& seg = segmentsInput[i];

            for (size_t j = 0; j < seg.size() - 1; j++) {
                size_t k = j + 1;

                cv::Point p1 = seg[j], p2 = seg[k];
                double dis = getDistancePTP(p1, p2);

                if (lenMax < dis) {
                    lenMax = dis;
                    iMax = i;
                    jMax = j;
                    pMax1 = p1;
                    pMax2 = p2;
                }
            }
        }

        if (lenMax < lenAvg) {
            std::cout << "[addVertices1] break: (lenMax<lenAvg) " << std::endl;
            std::cout << "[addVertices1] verticesNum: " << verticesNum << std::endl;
            return;
        }

        if (lenMax < 5.0) {
            std::cout << "[addVertices1] break: (lenMax<5.0) " << std::endl;
            std::cout << "[addVertices1] verticesNum: " << verticesNum << std::endl;
            return;
        }

        // insert new vertex
        cv::Point midp = (pMax1 + pMax2) / 2;

        auto& segMax = segmentsInput[iMax];
        auto segIt = segMax.begin() + jMax;
        segMax.insert(segIt + 1, midp);
        verticesNum++;

        std::cout << "[addVertices1] insert success: "
            << verticesNum << " "
            << midp << " "
            << pMax1 << " "
            << pMax2 << " "
            << iMax << " "
            << jMax << " "
            << std::endl;
    }

}

void edgeDrawingRun(const cv::Mat& imageInput, cv::Mat& imageResult, vector<vector<Point> >& segmentsResult) {

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

    // first process of vertex number
    //addVertices1(segments);

    // draw segments to result mat
    for (size_t i = 0; i < segments.size(); i++)
    {
        const Point* pts = &segments[i][0];
        int n = (int)segments[i].size();
        //random color
        polylines(edge_image_ed, &pts, &n, 1, false, Scalar((rand() & 255), (rand() & 255), (rand() & 255)), 1);
    }

    // draw circle for vertices
    for (size_t i = 0; i < segments.size(); i++)
    {
        auto& seg = segments[i];
        for (size_t j = 0; j < seg.size(); j++) {
            cv::Point p1 = seg[j];

            cv::circle(edge_image_ed, p1, 3, { 255,255,255 });
        }
    }


    imageResult = edge_image_ed;
    segmentsResult = segments;
}

void rasterization(const cv::Mat& imageInput, TriangulationMesh& triangulationMesh, bool debug_flag, std::string imName) {

    int _cnt1 = 0;

    cv::Mat imageResult = cv::Mat::zeros(imageInput.size(), CV_8UC3);

    double energySum = 0.0;

    for (auto& triPtr : triangulationMesh.trianglesList) {
        triPtr->updatePixels(imageInput, triangulationMesh);
        // solve
        triPtr->solve(imageInput);
        triPtr->energy = 0.0;
        for (auto& coordinatePair : triPtr->pixelsList)
        {
            auto x = (coordinatePair.first);
            auto y = (coordinatePair.second);
            auto interpolationResult = triPtr->getInterpolationResult(cv::Point2d(x, y));

            imageResult.at<cv::Vec3b>(y, x) = interpolationResult;

            // energySum update
            auto originalColor = imageInput.at<cv::Vec3b>(y, x);

            for (uint channelId = 0; channelId < 3; channelId++) {
                double hx = originalColor[channelId] / 256.0;
                double px = interpolationResult[channelId] / 256.0;

                double tmpEnergy = std::pow((hx - px), 2);

                energySum += tmpEnergy;
                triPtr->energy += tmpEnergy;
            }
        }


        _cnt1++;
        std::cout << "[rasterization] _cnt1: " << _cnt1 << " " << triangulationMesh.trianglesList.size() << std::endl;
    }
    double rmse = std::sqrt(energySum / (imageInput.size().width * imageInput.size().height));

    energy_vec.emplace_back(energySum);
    rmse_vec.emplace_back(rmse);
    std::cout << "[rasterization] energySum: " << energySum << std::endl;
    std::cout << "[rasterization] rmse: " << rmse << std::endl;

    imShowAndimWrite(imageResult, imName);
}

void energyRecal(
    const cv::Mat& imageInput,
    TriangulationMesh& triangulationMesh,
    std::vector<std::shared_ptr<TriangulationMesh::CurvedTriangle>> tmpTriList
)
{

    // re-calculate (almost same as rasterization)
    for (auto& triPtr : tmpTriList) {

        triPtr->updatePixels(imageInput, triangulationMesh);
        triPtr->solve(imageInput);

    }
}

// tmpTriList is unordered
double vertexEnergy(
    const cv::Mat& imageInput,
    TriangulationMesh& triangulationMesh,
    std::vector<std::shared_ptr<TriangulationMesh::CurvedTriangle>> tmpTriList,
    bool recalFlag
)
{
    if (recalFlag) {
        energyRecal(imageInput, triangulationMesh, tmpTriList);
    }

    double energySum = 0.0;
    // energy for every triangle
    for (auto& triPtr : tmpTriList) {
        triPtr->energy = 0.0;

        for (auto& coordinatePair : triPtr->pixelsList)
        {
            auto x = coordinatePair.first;
            auto y = coordinatePair.second;
            auto interpolationResult = triPtr->getInterpolationResult(cv::Point2d(x, y));

            for (uint channelId = 0; channelId < 3; channelId++) {

                double hx = imageInput.at<cv::Vec3b>(y, x)[channelId] / 256.0;
                double e = std::pow((hx - interpolationResult[channelId] / 256.0), 2);
                energySum += e;
                triPtr->energy += e;
            }
        }
    }

    return energySum;

}

void vertexGradient(const cv::Mat& imageInput, TriangulationMesh& triangulationMesh) {

    const double deltaT = 0.001;
    uint success_cnt = 0;

    for (size_t ii = 0; ii < triangulationMesh.pointsList.size(); ii++) {
        auto pointPtr = triangulationMesh.pointsList[ii];
        cv::Point2d cvpi(pointPtr->coordinate.first, pointPtr->coordinate.second);
        double safety_dis = MAXDIS;
        std::vector<std::shared_ptr<TriangulationMesh::CurvedTriangle>> triList; //unordered

        // check null triangle (temp)
        bool checkNullTriangleFlag = false;
        for (auto& edgePtr : pointPtr->incidentEdgesList) {
            if (edgePtr->leftTriangle == nullptr || edgePtr->rightTriangle == nullptr) {
                std::cout << "[vertexGradient] skipped, ii:" << ii << std::endl;
                checkNullTriangleFlag = true;
                break;
            }
        }
        if (checkNullTriangleFlag) continue;

        // calculate
        cv::Point2d sumGradient(0.0, 0.0);
        HalfSurfaceChecker halfSurfaceChecker;


        for (auto& edgePtr : pointPtr->incidentEdgesList) {
            auto leftTriPtr = edgePtr->leftTriangle;
            auto rightTriPtr = edgePtr->rightTriangle;


            auto otherPointIndex = edgePtr->vertexIndexPair.second;
            if (edgePtr->vertexIndexPair.first != ii) {
                std::swap(leftTriPtr, rightTriPtr);
                otherPointIndex = edgePtr->vertexIndexPair.first;
            }

            auto otherPointPtr = triangulationMesh.pointsList[otherPointIndex];

            // update triList (unordered)
            triList.emplace_back(leftTriPtr);

            // construct CV point (for easy calculation)
            cv::Point2d cvp2(otherPointPtr->coordinate.first, otherPointPtr->coordinate.second);
            cv::Point2d cvc1(edgePtr->controlPoint.first, edgePtr->controlPoint.second);

            // update halfSurfaceChecker
            halfSurfaceChecker.segmentsList.emplace_back(
                std::make_pair(
                    cvp2,
                    cvc1
                )
            );

            for (double t = 0.0; t <= 1.0; t += deltaT) {

                // bezier
                auto cvx = std::pow(1 - t, 2) * cvpi + std::pow(t, 2) * cvp2 + 2 * t * (1 - t) * cvc1;

                // deltax
                double deltax = 0.0;
                auto leftInterpolationResult = leftTriPtr->getInterpolationResult(cvx);
                auto rightInterpolationResult = rightTriPtr->getInterpolationResult(cvx);

                for (uint channelId = 0; channelId < 3; channelId++) {
                    double hx = imageInput.at<cv::Vec3b>(cvx.y, cvx.x)[channelId] / 256.0;
                    deltax += std::pow((hx - leftInterpolationResult[channelId] / 256.0), 2);
                    deltax -= std::pow((hx - rightInterpolationResult[channelId] / 256.0), 2);
                }

                deltax *= (1 - t) * (1 - t) * deltaT;

                // gradient direction
                auto dir = -2 * (1 - t) * cvpi + (2 - 4 * t) * cvc1 + 2 * t * cvp2;
                cv::Point2d dir2(dir.y, -dir.x); // rotate 270 (rotate 90 clockwise)

                sumGradient += dir2 * deltax;
            }
        }

        if (sumGradient == cv::Point2d(0.0, 0.0)) {
            std::cout << "[vertexGradient] sumGradient == 0, ii:" << ii << endl;
            continue;
        }

        sumGradient /= cv::norm(sumGradient);

        // half check


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
            //safety_dis = std::min(safety_dis, getDistancePTP(cvpi, cv::Point2d(cCoord.first, cCoord.second)));
        }

        // get max dis can move for half surface
        cv::Point2d cvpi_tmp = cvpi - sumGradient * safety_dis;
        while (halfSurfaceChecker.check(cvpi, cvpi_tmp) != 1 && (safety_dis > 0.001)) {
            safety_dis /= 2.0;
            cvpi_tmp = cvpi - sumGradient * safety_dis;
        }

        if (safety_dis <= 0.001) {
            std::cout << "[vertexGradient] safety_dis <= 0.001, update skipped, ii:" << ii << endl;
            continue;
        }

        // do gradient decrease
        double energyOrigin = vertexEnergy(imageInput, triangulationMesh, triList, false);
        const double CCC = 0.2;
        const uint TRY_TIMES = 5;
        const uint energyZeroCnt_UPBOUND = 2;

        // TODO: reverse search here??

        double alpha_star = CCC * safety_dis;
        cv::Point2d cvpi_new = cvpi;
        uint ti = 0;
        uint energyZeroCnt = 0;
        for (; ti < TRY_TIMES; ti++) {
            cvpi_new = cvpi - alpha_star * sumGradient;


            if (halfSurfaceChecker.check(cvpi, cvpi_new) != 1) {

                std::cout << "[controlPointGradient] halfSurfaceChecker failed: "
                    << ti << " "
                    << cvpi_new.x << " " << cvpi_new.y << " <- "
                    << cvpi.x << " " << cvpi.y
                    << std::endl;

                alpha_star *= CCC;
                continue;
            }

            // check energy decrease
            // set new cvpi_new to mesh
            pointPtr->setCoordinate(cvpi_new);
            double energyNew = vertexEnergy(imageInput, triangulationMesh, triList, true);
            double energyDelta = energyNew - energyOrigin;

            std::cout << "[vertexGradient] energy delta:" << (energyDelta) << "=" << (energyNew) << "-" << (energyOrigin) << std::endl;

            if (energyDelta < 0.0) {
                std::cout << "[vertexGradient] update success:"
                    << ti << " "
                    << cvpi_new.x << " " << cvpi_new.y << " <- "
                    << cvpi.x << " " << cvpi.y
                    << std::endl;

                // accept this set and break;
                success_cnt++;
                break;
            }
            else if (energyDelta == 0.0) {
                energyZeroCnt++;
                if (energyZeroCnt == energyZeroCnt_UPBOUND) {
                    break;
                }
            }

            alpha_star *= CCC;
        }

        if (ti == TRY_TIMES || energyZeroCnt == energyZeroCnt_UPBOUND) {
            // undo set cvci_new
            pointPtr->setCoordinate(cvpi);

            vertexEnergy(imageInput, triangulationMesh, triList, true);

            std::cout << "[vertexGradient] update failed:"
                << ti << " "
                << cvpi_new.x << " " << cvpi_new.y << " <- "
                << cvpi.x << " " << cvpi.y
                << std::endl;
        }

        std::cout << "[vertexGradient] Finish ii: " << ii << " " << triangulationMesh.pointsList.size() << std::endl;

    }

    std::cout << "[vertexGradient] All Finished, success_cnt: " << success_cnt << std::endl;
    success_cnt_vertex.emplace_back(success_cnt);
}


double controlPointEnergy(
    const cv::Mat& imageInput,
    TriangulationMesh& triangulationMesh,
    std::shared_ptr<TriangulationMesh::CurvedTriangle> leftTriPtr,
    std::shared_ptr<TriangulationMesh::CurvedTriangle> rightTriPtr,
    bool recalFlag
)
{
    const std::vector<std::shared_ptr<TriangulationMesh::CurvedTriangle>> tmpTriList{ leftTriPtr , rightTriPtr };

    if (recalFlag) {
        energyRecal(imageInput, triangulationMesh, tmpTriList);
    }

    double energySum = 0.0;

    // energy for left and right triangle
    for (auto& triPtr : tmpTriList) {
        triPtr->energy = 0.0;
        for (auto& coordinatePair : triPtr->pixelsList)
        {
            auto x = coordinatePair.first;
            auto y = coordinatePair.second;
            auto interpolationResult = triPtr->getInterpolationResult(cv::Point2d(x, y));

            for (uint channelId = 0; channelId < 3; channelId++) {

                double hx = imageInput.at<cv::Vec3b>(y, x)[channelId] / 256.0;
                double e = std::pow((hx - interpolationResult[channelId] / 256.0), 2);
                energySum += e;
                triPtr->energy += e;
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
            std::cout << "[controlPointGradient] nullptr skipped, ii:" << ii++ << std::endl;
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
            auto leftInterpolationResult = leftTriPtr->getInterpolationResult(cvx);
            auto rightInterpolationResult = rightTriPtr->getInterpolationResult(cvx);
            for (uint channelId = 0; channelId < 3; channelId++) {
                double hx = imageInput.at<cv::Vec3b>(cvx.y, cvx.x)[channelId] / 256.0;
                deltax += std::pow((hx - leftInterpolationResult[channelId] / 256.0), 2);
                deltax -= std::pow((hx - rightInterpolationResult[channelId] / 256.0), 2);
            }

            deltax *= 2 * t * (1 - t) * deltaT;
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
            uint kIndex = triangulationMesh.pointsList.size() + 100;
            for (auto index : triPtr->verticesIndexList) {
                if (index != p1Index && index != p2Index) {
                    kIndex = index;
                    break;
                }
            }

            auto kPtr = triangulationMesh.pointsList[kIndex];
            cv::Point2d cvpk(kPtr->coordinate.first, kPtr->coordinate.second);

            halfSurfaceChecker.segmentsList.emplace_back(
                std::make_pair(cvp1, cvpk)
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
        while (halfSurfaceChecker.check(cvc1, cvc1_tmp) != 1 && (safety_dis > 0.001)) {
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

        // TODO: reverse search here????

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

            std::cout << "[controlPointGradient] energy delta:" << (energyDelta) << "=" << (energyNew) << "-" << (energyOrigin) << std::endl;


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
            else if (energyDelta == 0.0) {
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

    std::cout << "[controlPointGradient] All Finished, success_cnt: " << success_cnt << std::endl;
    success_cnt_controlPoint.emplace_back(success_cnt);
}


void printStats() {
    std::cout << "success_cnt_controlPoint:";
    for (auto& cnt : success_cnt_controlPoint) {
        std::cout << cnt << " ";
    }
    std::cout << std::endl;

    std::cout << "success_cnt_vertex:";
    for (auto& cnt : success_cnt_vertex) {
        std::cout << cnt << " ";
    }
    std::cout << std::endl;

    std::cout << "energySum:";
    for (auto& e : energy_vec) {
        std::cout << e << " ";
    }
    std::cout << std::endl;

    std::cout << "rmse:";
    for (auto& r : rmse_vec) {
        std::cout << r << " ";
    }
    std::cout << std::endl;


}

int main2()
{
    String imgPath = "./siyecao.jpg"; //siyecao


    Mat image;
    image = imread(imgPath, IMREAD_COLOR); // Read the file
    if (image.empty()) // Check for invalid input
    {
        std::cout << "[main2] Could not open or find the image" << std::endl;
        return -1;
    }

    // process
    {
        std::cout << "[main2] COLS:" << image.cols << endl;
        std::cout << "[main2] ROWS:" << image.rows << endl;
        std::cout << "[main2] CHANNELS:" << image.channels() << endl;

        // 1. edge drawing
        cv::Mat result_EdgeDrawing_mat;
        vector<vector<Point> > result_EdgeDrawing_segments;
        edgeDrawingRun(image, result_EdgeDrawing_mat, result_EdgeDrawing_segments);

        std::string edgeDrawing_imName = "[main2] EdgeDrawing detected edges.jpg";
        imShowAndimWrite(result_EdgeDrawing_mat, edgeDrawing_imName);

        // 2. triangulation
        cv::Mat result_Triangulation_mat;
        TriangulationMesh result_triangulationMesh;
        TriangulationRun(image, result_EdgeDrawing_segments, result_Triangulation_mat, result_triangulationMesh);

        std::string triangulation_imName = "[main2] TriangulationImage.jpg";
        imShowAndimWrite(result_Triangulation_mat, triangulation_imName);


        // 3. rasterization
        rasterization(image, result_triangulationMesh, false, "[rasterization] imageResult.jpg");

        // 3.1 addVertices2
        addVertices2(image, result_triangulationMesh);

        // 3.2 rasterization again

        cv::Mat result_Triangulation_mat_addVertices2;
        drawMesh(image, result_triangulationMesh, result_Triangulation_mat_addVertices2);
        imShowAndimWrite(result_Triangulation_mat_addVertices2, "[main2] TriangulationImage_addVertices2.jpg");
        rasterization(image, result_triangulationMesh, false, "[rasterization] imageResult_addVertices2.jpg");


        const uint it_i_BOUND = 10;
        for (uint it_i = 0; it_i < it_i_BOUND; it_i++)
        {
            cout << "[main2] UPDATE it_i:" << it_i << endl;

            // 4. control point update
            {
                controlPointGradient(image, result_triangulationMesh);
                cv::Mat result_Triangulation_mat3;
                drawMesh(image, result_triangulationMesh, result_Triangulation_mat3);

                std::string controlPoint_imName = "[main2] controlPointGradient Image";
                controlPoint_imName += std::to_string(it_i);
                controlPoint_imName += ".jpg";
                imShowAndimWrite(result_Triangulation_mat3, controlPoint_imName);
            }

            // 5. vetices update
            {
                vertexGradient(image, result_triangulationMesh);
                cv::Mat result_Triangulation_mat2;
                drawMesh(image, result_triangulationMesh, result_Triangulation_mat2);

                std::string vertex_imName = "[main2] vertexGradient Image";
                vertex_imName += std::to_string(it_i);
                vertex_imName += ".jpg";
                imShowAndimWrite(result_Triangulation_mat2, vertex_imName);
            }


            // 6. result
            {
                string res_imName = "[rasterization] imageResult";
                res_imName += std::to_string(it_i);
                res_imName += ".jpg";
                rasterization(image, result_triangulationMesh, false, res_imName);
            }

        }

        // print stats
        printStats();


        wt(1);
    }


    return 0;
}


int _test() {
    cv::Point c(2, 2), a(4, -3), b(6, 2);
    std::cout << getDistancePTL(a, b, c) << endl;

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
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);

    //return _test2();
    return main2();
    //return _test();
    return 0;
}