#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

double t_a = 0.95;
double sigma_w = 2.25;

Mat DoG_convolution(Mat src)
{
    Mat g1, g2, result, image;
    GaussianBlur(src, g1, Size(3, 3), 0.5);
    GaussianBlur(src, g2, Size(3, 3), 1.5);
    result = g1 - g2;
    image = src + result;
    return image;
}

double mean(Mat input){
    double sum = 0;
    int count = 0;
    for(int i=0 ; i<input.rows ; i++){
        double* input_data = input.ptr<double>(i);
        for(int j=0 ; j<input.cols ; j++){
            if(input_data[j] != 0){
                count++;
                sum += input_data[j];
            }
        }
    }
    return sum / count;
}

Mat exp_function(Mat input)
{
    Mat exp_input(input.size(), input.type());
    exp_input = (input ^ 2 / 0.5);
    Mat dst(input.size(), input.type());
    exp(exp_input, dst);
    return dst;
}

double amb_argmin(Mat input, Mat recursive, Mat Value, double r){
    Mat dst(input.size(), input.type());
    pow(input,r,dst);
    double up = mean(dst) - Value.at<double>(0,0);
    for(int i =0; i<dst.rows; i++){
        double* dst_data = dst.ptr<double>(i);
        double* recursive_data = recursive.ptr<double>(i);
        for(int j=0; j<dst.cols; j++){
            if(dst_data[j] != 0){
                dst_data[j] = dst_data[j]*log(recursive_data[j]);
            }else{
                dst_data[j] = 0;
            }
        }
    }
    
    double down = mean(dst);
    double optimal_r = r - (up/down);
    if(r - optimal_r<=0.0000001){
        return optimal_r;
    }else{
        amb_argmin(input, recursive, Value, optimal_r);
    }
}

Mat OGGCPE(Mat src)
{
    Mat channel[3];
    split(src, channel);
    double max = 0;
    double min = 0;

    Mat L_in = Mat::zeros(Size(src.rows, src.cols), CV_8UC1);
    L_in = channel[2] * 0.299 + 0.587 * channel[1] + 0.114 * channel[0];

    minMaxLoc(L_in, &min, &max);
    Mat bright_set(src.size(), CV_64FC1);
    Mat dark_set(src.size(), CV_64FC1);
    Mat L_log(src.size(), CV_64FC1);
    
    for (int i = 0; i < src.rows; i++)
    {
        uchar* L_in_data  = L_in.ptr<uchar>(i);
        double* L_log_data = L_log.ptr<double>(i);
        for (int j = 0; j < src.cols; j++)
        {
            L_log_data[j] = log1p(L_in_data[j]) / log1p(max);
        }
    }

    for (int i = 0; i < src.rows; i++){
        double* L_log_data = L_log.ptr<double>(i);
        double* bright_set_data = bright_set.ptr<double>(i);
        double* dark_set_data = dark_set.ptr<double>(i);
        for (int j = 0; j < src.cols; j++){
            if(L_log_data[j] > 0.5){
                bright_set_data[j] = L_log_data[j];
            }else{
                dark_set_data[j] = L_log_data[j];
            }
        }
    }


    Mat mean_low, stddev_low, Mat_Low;
    Mat mean_high, stddev_high, Mat_High;
    meanStdDev(dark_set, mean_low, stddev_low);
    meanStdDev(bright_set, mean_high, stddev_high);

    Mat_Low =  stddev_low;
    Mat_High =  1 - stddev_high;

    double gamma_low = amb_argmin(dark_set,dark_set,Mat_Low, 1.0);
    double gamma_high = amb_argmin(bright_set,bright_set,Mat_High, 1.0);
    
    Mat L_dark(dark_set.size(), dark_set.type());
    Mat L_bright(bright_set.size(), bright_set.type());

    pow(L_log, gamma_low, L_dark);
    pow(L_log, gamma_high, L_bright);

    Mat comma_dark = DoG_convolution(L_dark);
    Mat comma_bright = DoG_convolution(L_bright);

    Mat w(L_bright.size(), L_bright.type());
    Mat L_pow(L_bright.size(), L_bright.type());

    pow(L_bright, 3.0, L_pow);
    exp(sigma_w * L_pow, w);

    Mat l_out(w.size(), w.type());
    for (int x = 0; x < l_out.rows; x++)
    {
        double* l_out_data = l_out.ptr<double>(x);
        double* w_data = w.ptr<double>(x);
        double* comma_dark_data = comma_dark.ptr<double>(x);
        double* comma_bright_data = comma_bright.ptr<double>(x);
        for (int y = 0; y < l_out.cols; y++)
        {
            l_out_data[y] = w_data[y] * comma_dark_data[y] + (1 - w_data[y]) * comma_bright_data[y];
        }
    }
    Mat s(L_bright.size(), L_bright.type());
    
    for (int x = 0; x < L_bright.rows; x++)
    {
        double* s_data = s.ptr<double>(x);
        double* L_bright_data = L_bright.ptr<double>(x);
        for (int y = 0; y < L_bright.cols; y++)
        {
            s_data[y] = t_a - tanh(L_bright_data[y]);
        }
    }

    Mat I_c[3] = Mat(src.size(), CV_64FC1);
    L_in.convertTo(L_in, CV_64F);
    L_in = L_in / 255.0;
    
    Mat test_I_c[3];
    test_I_c[0] = L_in;
    test_I_c[1] = L_in;
    test_I_c[2] = L_in;

    Mat result_I_c = Mat(src.size(), CV_64FC3);
    merge(test_I_c,3,result_I_c);
    
    Mat src_test = Mat(src.size(), CV_64FC3);
    src.convertTo(src,CV_64FC3);
    src = src / 255;
    
    divide(src,result_I_c,src_test,1.0,-1);
    split(src_test,I_c);
    
    Mat result[3];
    result[0] = Mat(src.size(), CV_64F);
    result[1] = Mat(src.size(), CV_64F);
    result[2] = Mat(src.size(), CV_64F);
    for (int x = 0; x < L_in.rows; x++)
    {
        double* I_c_0_data = I_c[0].ptr<double>(x);
        double* I_c_1_data = I_c[1].ptr<double>(x);
        double* I_c_2_data = I_c[2].ptr<double>(x);
        double* s_data = s.ptr<double>(x);
        for (int y = 0; y < L_in.cols; y++)
        {
            I_c_0_data[y] = pow(I_c_0_data[y], s_data[y]);
            I_c_1_data[y] = pow(I_c_1_data[y], s_data[y]);
            I_c_2_data[y] = pow(I_c_2_data[y], s_data[y]);
        }
    }
    Mat result_I = Mat(src.size(), CV_64FC3);
    merge(I_c,3,result_I);
    Mat dst = Mat(src.size(), CV_64FC3);
    Mat l_out_test[3];
    l_out_test[0] = l_out;
    l_out_test[1] = l_out;
    l_out_test[2] = l_out;
    Mat l_out_result = Mat(src.size(), CV_64FC3);
    merge(l_out_test,3,l_out_result);
    multiply(l_out_result,result_I,dst,1.0,-1);
    dst = dst * 255 ;
    dst.convertTo(dst, CV_8UC3);
    return dst;
}
int main()
{
    Mat src = imread("/home/cilab/Desktop/B_9.png", 1);
    CV_Assert(src.data);
    Mat result = OGGCPE(src);
    imshow("result", result);
    waitKey(0);
    return 0;
}
