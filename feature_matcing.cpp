#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using std::cout;
using std::endl;

const char* keys = 
"{ help h | | Print help message. }"
"{ input1 | ../data/box.png | Path to input image 1. }"
"{ input2 | ../data/box_in_scene.png | Path to input image 2. }";

int main( int argc, char* argv[] ) {
    CommandLineParser parser( argc, argv, keys );
    Mat img1 = imread( samples::findFile( parser.get<String>("input1") ), IMREAD_GRAYSCALE );
    Mat img2 = imread( samples::findFile( parser.get<String>("input2") ), IMREAD_GRAYSCALE );

    if ( img1.empty() || img2.empty() ) {
        cout << "Could not open or find the image!" << endl;
        cout << "Usage: " << argv[0] << " <Input image 1> <Input image 2>" << endl;
        return -1;
    }

    int minHessian = 400;

    Ptr<SIFT> detector = SIFT::create( minHessian );
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( DescriptorMatcher::BRUTEFORCE );
    std::vector< DMatch > matches;
    matcher->match( descriptors1, descriptors2, matches );

    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, matches, img_matches );
    

    imshow( "Matches", img_matches );
    waitKey();

    return 0;
}