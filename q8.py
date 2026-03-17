#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cmath>

using namespace std;
using namespace cv;

// -------------------- BLUR --------------------
Mat blurImage(Mat input) {
    Mat output = input.clone();

    #pragma omp parallel for
    for(int i = 1; i < input.rows - 1; i++) {
        for(int j = 1; j < input.cols - 1; j++) {
            for(int c = 0; c < 3; c++) {

                int sum = 0;
                for(int ki = -1; ki <= 1; ki++) {
                    for(int kj = -1; kj <= 1; kj++) {
                        sum += input.at<Vec3b>(i+ki, j+kj)[c];
                    }
                }

                output.at<Vec3b>(i,j)[c] = sum / 9;
            }
        }
    }

    return output;
}

// -------------------- SHARPEN --------------------
Mat sharpenImage(Mat input) {
    Mat output = input.clone();

    int kernel[3][3] = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    };

    #pragma omp parallel for
    for(int i = 1; i < input.rows - 1; i++) {
        for(int j = 1; j < input.cols - 1; j++) {
            for(int c = 0; c < 3; c++) {

                int sum = 0;
                for(int ki = -1; ki <= 1; ki++) {
                    for(int kj = -1; kj <= 1; kj++) {
                        sum += input.at<Vec3b>(i+ki, j+kj)[c] *
                               kernel[ki+1][kj+1];
                    }
                }

                output.at<Vec3b>(i,j)[c] =
                    saturate_cast<uchar>(sum);
            }
        }
    }

    return output;
}

// -------------------- SOBEL EDGE --------------------
Mat sobelEdge(Mat input) {

    Mat gray, output;
    cvtColor(input, gray, COLOR_BGR2GRAY);
    output = gray.clone();

    int Gx[3][3] = {
        {-1,0,1}, {-2,0,2}, {-1,0,1}
    };

    int Gy[3][3] = {
        {-1,-2,-1}, {0,0,0}, {1,2,1}
    };

    #pragma omp parallel for
    for(int i = 1; i < gray.rows - 1; i++) {
        for(int j = 1; j < gray.cols - 1; j++) {

            int sumX = 0, sumY = 0;

            for(int ki = -1; ki <= 1; ki++) {
                for(int kj = -1; kj <= 1; kj++) {
                    sumX += gray.at<uchar>(i+ki,j+kj) * Gx[ki+1][kj+1];
                    sumY += gray.at<uchar>(i+ki,j+kj) * Gy[ki+1][kj+1];
                }
            }

            int magnitude = sqrt(sumX*sumX + sumY*sumY);
            output.at<uchar>(i,j) =
                saturate_cast<uchar>(magnitude);
        }
    }

    return output;
}

// -------------------- MAIN --------------------
int main() {

    // Load image
    Mat image = imread("input.jpg");

    if(image.empty()) {
        cout << "Error loading image!\n";
        return -1;
    }

    cout << "Number of threads: " << omp_get_max_threads() << endl;

    imshow("Original Image", image);

    // -------- BLUR --------
    double start = omp_get_wtime();
    Mat blurImg = blurImage(image);
    double end = omp_get_wtime();
    cout << "Blur Time: " << end - start << " seconds\n";

    // -------- SHARPEN --------
    start = omp_get_wtime();
    Mat sharpImg = sharpenImage(image);
    end = omp_get_wtime();
    cout << "Sharpen Time: " << end - start << " seconds\n";

    // -------- SOBEL --------
    start = omp_get_wtime();
    Mat edgeImg = sobelEdge(image);
    end = omp_get_wtime();
    cout << "Sobel Time: " << end - start << " seconds\n";

    // Display results
    imshow("Blur", blurImg);
    imshow("Sharpen", sharpImg);
    imshow("Edge", edgeImg);

    // Save results
    imwrite("blur.jpg", blurImg);
    imwrite("sharpen.jpg", sharpImg);
    imwrite("edge.jpg", edgeImg);

    waitKey(0);
    destroyAllWindows();

    return 0;
    
}
