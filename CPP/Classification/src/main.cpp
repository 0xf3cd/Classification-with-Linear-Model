#include "Input.h"
#include "LogisticRegression.h"
#include "tabulate/table.hpp"
#include "Dataset.h"
#include "DataTransformer.h"
#include "Classifier.h"
#include <random>
#include <vector>
#include <string>
#include <iostream>

#ifdef DATASET_IRIS
    #define DATASET_DIR "../dataset/iris"
#else 
    #define DATASET_DIR "../dataset/heart"
#endif

typedef std::vector<variant<std::string, const char *, tabulate::Table> > Row;

template<typename Type, typename... Types>
void print(const Type& arg, const Types&... args);

tabulate::Table genPredTable(std::vector<std::string> &pred,
                             std::vector<Prediction> &pred_detail,
                             std::vector<std::string> &actual,
                             std::vector<std::string> &all_classes);

// return the index of the element with max value
int getMaxProbIdx(Prediction &IP);

int main() {
    const double DATASET_SPLIT_RATE = 0.9; // there will be (DATASET_SIZE*DATASET_SPLIT_RATE) items in the training set

    DataTransformer DT;
    DT.init(DATASET_DIR, DATASET_SPLIT_RATE);
    auto all_classes = DT.getAllClasses();
    auto all_DS = DT.getAllDS();
    TEST_SET_TYPE test_set_tuple = DT.getTestSet();

    print("------------------------------------------------\n\n");

    #ifdef DATASET_IRIS
        print("Dataset used: ", "Iris", '\n');
    #else 
        print("Dataset used: ", "HeartAttack", '\n');
    #endif

    #ifdef USE_LINEAR
        print("Model used: ", "Linear Regression", '\n');
    #else 
        print("Model used: ", "Logistic Regression", '\n');
    #endif

    print("Dataset read-in successfully.\n");
    print("Training set size: ", DT.getTrainSetSize(), '\n');
    print("Test set size: ", DT.getTestSetSize(), '\n');

    print("\n------------------------------------------------\n\n");

    Classifier C(all_classes, all_DS);

    print("Training started.\n");
    auto train_details = C.train();
    print("Training finished.\n");

    int count = 0;
    for(const auto& train_detail : train_details) {
        print('\n', "sub-classifier for class ", all_classes[count++], '\n');

        tabulate::Table train_table; 
        train_table.add_row({"Iterations", "Initial Cost Value", "Final Cost Value"}); 
        train_table[0][0].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
        train_table[0][1].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
        train_table[0][2].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::green);
        train_table.add_row({std::to_string(std::get<0>(train_detail)),
                            std::to_string(std::get<1>(train_detail)),
                            std::to_string(std::get<2>(train_detail))});

        print(train_table, '\n');
    }

    print("\n------------------------------------------------\n\n");

    print("Use trained model to perform prediction on test set.\n");
    auto pred = C.predict(std::get<0>(test_set_tuple));
    auto pred_detail = C.predictWithDetail(std::get<0>(test_set_tuple));
    auto actual = std::get<1>(test_set_tuple);
    
    auto predict_table = genPredTable(pred, pred_detail, actual, all_classes);
    print(predict_table, '\n');

    return 0;
}

void print() {} 
 
template<typename Type, typename... Types>
void print(const Type& arg, const Types&... args) {
    std::cout << arg;
    print(args...);
}

tabulate::Table genPredTable(std::vector<std::string> &pred,
                             std::vector<Prediction> &pred_detail,
                             std::vector<std::string> &actual,
                             std::vector<std::string> &all_classes) 
{
    const int class_amount = all_classes.size();

    tabulate::Table predict_table; 

    // prepare for the first row of the table
    Row first_row {"No.", "Estimated Class", "Real Class", "Correct?"};
    for(int i = 0; i < class_amount; i++) {
        auto iter = first_row.begin() + 1;
        first_row.insert(iter, all_classes[i]+" Probability");
    }
    predict_table.add_row(first_row); 

    // format the first row
    for(int i = 0; i < 4+class_amount; i++) {
        predict_table[0][i].format()
                           .font_style({tabulate::FontStyle::bold})
                           .font_color(tabulate::Color::green);
    }

    // prepare for the next rows
    for(int i = 0; i < actual.size(); i++) {
        const bool is_correct = pred[i] == actual[i];
        Row row {std::to_string(i+1)};
        Row row_end { pred[i], actual[i], is_correct? "Yes" : "No"};

        for(int j = 0; j < class_amount; j++) {
            row.insert(std::end(row), std::to_string(pred_detail[i][all_classes[j]]));
        }
        row.insert(std::end(row), std::begin(row_end), std::end(row_end)); // append row_end to row

        predict_table.add_row(row);

        if(is_correct) {
            predict_table[i+1][row.size()-1].format().font_color(tabulate::Color::green);
        } else {
            predict_table[i+1][row.size()-1].format().font_color(tabulate::Color::red);
        }

        const int max_prox_idx = getMaxProbIdx(pred_detail[i]);
        predict_table[i+1][max_prox_idx+1].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::magenta);
    }

    return predict_table;
}

// return the index of the element with max value
int getMaxProbIdx(Prediction &IP) {
    std::vector<std::string> keys;
    std::vector<double> values;
    
    // split the IrisPrediction into 2 vectors
    // referring to the keys and values respectively
    for(const auto& each_pair : IP) {
        keys.push_back(each_pair.first);
        values.push_back(each_pair.second);
    }

    auto max_value_iter = std::max_element(values.begin(), values.end());
    return std::distance(values.begin(), max_value_iter);
}