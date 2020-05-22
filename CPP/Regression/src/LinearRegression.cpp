#include "LinearRegression.h"
#include <random>

/*
class LinearRegression {
private:
    Dataset DS;
    std::vector<double> coef; // coef for "coefficient"
    double bias;
    double alpha; 
    int iter; // how many iterations the training process will perform
};
*/

LinearRegression::LinearRegression(std::unique_ptr<Dataset> pDS) {
    this->pDS = std::move(pDS);
    this->alpha = 0.001;
    this->iter = 10000;
    this->cost_threshold = 0.0001;

    std::random_device rd;
    std::uniform_real_distribution<double> dist(-1, 1);

    (this->coef).clear();
    for(int i = 0; i < (*this->pDS).getInputSize(); i++) {
        (this->coef).push_back(dist(rd));
    }

    this->bias = dist(rd);
}

// perform prediction on one set of input
double LinearRegression::predictOnce(Input x) {
    const int input_size = x.size();
    const int coef_size = coef.size();

    if(input_size != coef_size) { // if x and coefficients mismatch
        return 0.0; 
    }

    double result = this->bias;
    for(int i = 0; i < input_size; i++) {
        result += x[i] * coef[i];
    }

    return result;
}

// calculate the cost value (namely the value of function J)
double LinearRegression::calcCostValue(std::vector<double> predicted_Y) {
    const int DS_size = (*this->pDS).getSize();
    std::vector<double> Y = (*this->pDS).getY();

    if(Y.size() != predicted_Y.size()) { // mismatch
        return -1.0;
    }

    double cost_value = 0.0;
    for(int i = 0; i < Y.size(); i++) {
        cost_value += (Y[i] - predicted_Y[i]) * (Y[i] - predicted_Y[i]);
    }
    cost_value /= 2 * DS_size;

    return cost_value;
}

void LinearRegression::setAlpha(double alpha) {
    this->alpha = alpha;
}

void LinearRegression::setIter(int iter) {
    this->iter = iter;
}

TrainRet LinearRegression::train() {
    const int DS_size = (*this->pDS).getSize();
    std::vector<Input> X = (*this->pDS).getX();
    std::vector<double> Y = (*this->pDS).getY();

    const double init_cost_value = this->calcCostValue(this->predict(X));
    double final_cost_value;
    int iter = 0;
    for(; iter < this->iter; iter++) {
        std::vector<double> predicted_Y = this->predict(X);

        const double cost_value = this->calcCostValue(predicted_Y);
        final_cost_value = cost_value;
        if(cost_value <= this->cost_threshold) {
            break; // when the cost value reaches the threshold, stop iteration
        }

        // sigma is an array, each element of which refers to each coefficient
        // std::unique_ptr<double[]> sigma(new double[(this->coef).size()]); 
        
        // calculate the value of sigma_bias, and then update the bias
        double sigma_bias = 0.0;
        for(int idx = 0; idx < DS_size; idx++) {
            sigma_bias += Y[idx] - predicted_Y[idx];
        }
        this->bias += this->alpha * sigma_bias / DS_size; // update the bias

        // perform gradient descent on the j-th coefficient
        for(int j = 0; j < (this->coef).size(); j++) {
            double sigma = 0.0;
            for(int idx = 0; idx < DS_size; idx++) {
                sigma += (Y[idx] - predicted_Y[idx]) * X[idx][j];
            }
            (this->coef)[j] += this->alpha * sigma / DS_size; // update the coefficient
        }
    }

    return std::make_tuple(iter, init_cost_value, final_cost_value);
}

// use the trained model to perform prediction on 
// items in X one by one
std::vector<double> LinearRegression::predict(std::vector<Input> X) {
    std::vector<double> Y;

    for(const auto& x : X) {
        Y.push_back(this->predictOnce(x));
    }

    return Y;
}

// return the coefficients
std::vector<double> LinearRegression::getCoef() {
    return std::vector<double>(this->coef);
}

double LinearRegression::getBias() {
    return this->bias;
}