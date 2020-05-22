#include "Input.h"
#include "LinearRegression.h"
#include "tabulate/table.hpp"
#include "Dataset.h"
#include <random>
#include <vector>
#include <string>
#include <iostream>

double getRandomBias(double min=-5.0, double max=5.0);
std::vector<double> getRandomCoef(int coef_size, double min=-5.0, double max=5.0);
std::unique_ptr<Dataset> getRandomDataset(int DS_size, int input_size, double bias, std::vector<double> coef, double min_x=-5.0, double max_x=5.0);


template<typename Type, typename... Types>
void print(const Type& arg, const Types&... args);


int main() {
    const int DATASET_SIZE = 200;
    const int COEF_SIZE = 10;
    const double DATASET_SPLIT_RATE = 0.9; // there will be (DATASET_SIZE*DATASET_SPLIT_RATE) items in the training set

    const double bias = getRandomBias();
    const std::vector<double> coef = getRandomCoef(COEF_SIZE);
    const std::unique_ptr<Dataset> pDS = getRandomDataset(DATASET_SIZE, COEF_SIZE, bias, coef);  

    // split the dataset into 2 parts
    std::unique_ptr<Dataset> pDS_train = (*pDS).split(0, DATASET_SIZE*DATASET_SPLIT_RATE-1);
    std::unique_ptr<Dataset> pDS_test = (*pDS).split(DATASET_SIZE*DATASET_SPLIT_RATE, DATASET_SIZE-1);

    print("------------------------------------------------\n\n");

    print("Dataset generated successfully.\n");
    print("Training set size: ", pDS_train->getSize(), '\n');
    print("Test set size: ", pDS_test->getSize(), '\n');

    print("\n------------------------------------------------\n\n");

    LinearRegression LR(std::move(pDS_train));
    // const std::vector<double> coef_before = LR.getCoef();
    // const double bias_before = LR.getBias();

    print("Training started.\n");
    auto train_detail = LR.train();
    print("Training finished.\n");

    tabulate::Table train_table; 
    train_table.add_row({"Iterations", "Initial Cost Value", "Final Cost Value"}); 
    train_table[0][0].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
    train_table[0][1].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
    train_table[0][2].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
    train_table.add_row({std::to_string(std::get<0>(train_detail)),
                         std::to_string(std::get<1>(train_detail)),
                         std::to_string(std::get<2>(train_detail))});

    print('\n', train_table, '\n');

    print("\n------------------------------------------------\n\n");

    const std::vector<double> coef_estimated = LR.getCoef();
    const double bias_estimated = LR.getBias();

    tabulate::Table bias_table; 
    bias_table.add_row({"Real Bias Value", "Estimated Bias Value"}); 
    bias_table[0][0].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
    bias_table[0][1].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
    bias_table.add_row({std::to_string(bias), std::to_string(bias_estimated)});

    print(bias_table, '\n');

    print("\n------------------------------------------------\n\n");

    tabulate::Table coef_table; 
    coef_table.add_row({"No.", "Real Coefficient Value", "Estimated Coefficient Value"}); 
    coef_table[0][0].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
    coef_table[0][1].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
    coef_table[0][2].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);

    for(int i = 0; i < COEF_SIZE; i++) {
        coef_table.add_row({std::to_string(i+1), std::to_string(coef[i]), std::to_string(coef_estimated[i])});
    }

    print(coef_table, '\n');

    print("\n------------------------------------------------\n\n");

    print("Use trained model to perform prediction on test set.\n");
    std::vector<double> predicted_Y = LR.predict(pDS_test->getX());
    std::vector<double> actual_Y = pDS_test->getY();

    tabulate::Table predict_table; 
    predict_table.add_row({"No.", "Real Value", "Estimated Value", "Diff"}); 
    predict_table[0][0].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
    predict_table[0][1].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
    predict_table[0][2].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
    predict_table[0][3].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);

    for(int i = 0; i < predicted_Y.size(); i++) {
        const double diff = abs(predicted_Y[i]-actual_Y[i]);
        predict_table.add_row({
            std::to_string(i+1), 
            std::to_string(predicted_Y[i]), 
            std::to_string(actual_Y[i]),
            std::to_string(diff)
        });

        if(diff > 0.02) {
            predict_table[i+1][3].format().font_color(tabulate::Color::red);
        } else {
            predict_table[i+1][3].format().font_color(tabulate::Color::green);
        }
    }

    print(predict_table, '\n');

    return 0;
}


double getRandomBias(double min, double max) {
    std::random_device rd;
    std::uniform_real_distribution<double> dist(min, max);

    return dist(rd);
}

std::vector<double> getRandomCoef(int coef_size, double min, double max) {
    std::random_device rd;
    std::uniform_real_distribution<double> dist(min, max);

    std::vector<double> coef;
    for(int i = 0; i < coef_size; i++) {
        coef.push_back(dist(rd));
    }

    return coef;
}

// coef.size() should be equal to input_size
std::unique_ptr<Dataset> getRandomDataset(int DS_size, int input_size, double bias, std::vector<double> coef, double min_x, double max_x) {
    if(input_size != coef.size()) {
        return std::unique_ptr<Dataset>(nullptr);
    }

    std::vector<Input> X;
    std::vector<double> Y;

    std::random_device rd;
    std::uniform_real_distribution<double> dist(min_x, max_x);

    for(int i = 0; i < DS_size; i++) {
        Input temp_x;
        for(int j = 0; j < input_size; j++) {
            temp_x.push_back(dist(rd));
        }
        X.push_back(temp_x);

        // calculate the corresponding value of y
        double temp_y = bias;
        for(int j = 0; j < input_size; j++) {
            temp_y += coef[j] * temp_x[j];
        }
        Y.push_back(temp_y);
    }

    std::unique_ptr<Dataset> pDS(new Dataset);
    pDS->init(X, Y);
    return std::move(pDS);
}

void print() {} 
 
template<typename Type, typename... Types>
void print(const Type& arg, const Types&... args) {
    std::cout << arg;
    print(args...);
}