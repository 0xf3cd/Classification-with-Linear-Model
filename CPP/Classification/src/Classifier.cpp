#include "Classifier.h"
#include <set>
#include <string>

/*
class Classifier {
private:
    std::vector<std::string> all_classes;
    MyMap map_class_LR; // one class corresponds to one dataset
*/

// make sure that the names of the classes are different from each other!
// class_names.size() should be equal to pDS_vector.size().
// each class has its own database, and the items in a database are marked with
// either "1" or "-1", where "1" means that the item belongs to the class
Classifier::Classifier(std::vector<std::string> &class_names, std::vector<std::unique_ptr<Dataset> > &pDS_vector) {
    const int class_amount = class_names.size();
    this->all_classes = std::vector<std::string>(class_names);

    for(int i = 0; i < class_amount; i++) {
        // insert a new pair into map_str_LR
        map_class_LR.insert(MyMap::value_type(class_names[i], ModelType(pDS_vector[i]->duplicate())));
    }
}

std::vector<TrainRet> Classifier::train() {
    std::vector<TrainRet> tr_vec;
    for(const auto& class_name : all_classes) {
        auto iter = map_class_LR.find(class_name);
        auto train_detail = (iter->second).train();
        tr_vec.push_back(train_detail);
    }

    return std::move(tr_vec);
}


// perform prediction on one item
// return the class that x is most likely to belong to
std::string Classifier::predictOnce(Input x) {
    Prediction IP = predictOnceWithDetail(x);
    auto first_element = IP.begin();
    double max_prob = first_element->second;
    std::string most_likely_class = first_element->first;

    for(const auto& each_elem : IP) {
        if(each_elem.second > max_prob) {
            max_prob = each_elem.second;
            most_likely_class = each_elem.first;
        }
    }

    return most_likely_class;
}

// perform prediction on one item
// return the probabilities that the item belongs to every class
Prediction Classifier::predictOnceWithDetail(Input x) {
    Prediction IP;
    for(const auto& class_name : all_classes) {
        auto iter = map_class_LR.find(class_name);
        double prob = (iter->second).predictOnce(x);
        IP.insert(std::pair<std::string, double>(class_name, prob));
    }

    return std::move(IP);
}

// perform predictOnce on each item in X
std::vector<std::string> Classifier::predict(std::vector<Input> X) {
    std::vector<std::string> prediction;

    for(const auto& x : X) {
        prediction.push_back(this->predictOnce(x));
    }

    return std::move(prediction);
}

// perform predictOnceWithDetail on each item in X 
std::vector<Prediction> Classifier::predictWithDetail(std::vector<Input> X) {
    std::vector<Prediction> prediction;

    for(const auto& x : X) {
        prediction.push_back(this->predictOnceWithDetail(x));
    }

    return std::move(prediction);
}