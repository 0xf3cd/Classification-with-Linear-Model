#include "Input.h"
#include "Dataset.h"
#include <vector>
#include <tuple>

typedef std::tuple<int, double, double> TrainRet;

class LinearRegression {
private:
    std::unique_ptr<Dataset> pDS;
    std::vector<double> coef; // coef for "coefficient"
    double bias;
    double alpha; 
    int iter; // how many iterations the training process will perform
    double cost_threshold; // when the cost value reaches the threshold, stop iteration

    // perform prediction on one item
    double predictOnce(Input x);

    // calculate the cost value (namely the value of function J)
    double calcCostValue(std::vector<double> predicted_Y);

public:
    LinearRegression(std::unique_ptr<Dataset> pDS);
    void setAlpha(double alpha);
    void setIter(int iter);
    TrainRet train();

    // use the trained model to perform prediction on 
    // items in X one by one
    std::vector<double> predict(std::vector<Input> X);

    // return the coefficients
    std::vector<double> getCoef();

    double getBias();
};