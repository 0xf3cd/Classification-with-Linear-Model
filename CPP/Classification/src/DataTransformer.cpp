#include "DataTransformer.h"
#include <algorithm>
#include <random>

/*
class DataTransformer {
private: 
    std::ifstream fin;
    std::vector<OriDataItem> all_ori_dataitems;
    std::vector<DataItem> all_dataitems;
    std::set<std::string> all_classes;
    std::map<std::string, int> map_class_no;
    std::map<int, std::string> map_no_class;
    std::unique_ptr<Dataset> test_set;
    std::unique_ptr<Dataset> train_set;
    std::map<std::string, std::unique_ptr<Dataset> > map_class_DS;
*/

// shuffle all_ori_dataitems
void DataTransformer::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());

    auto& items = this->all_ori_dataitems;
    std::shuffle(items.begin(), items.end(), g);
}

void DataTransformer::transformDataitems() {
    for(const auto& odi : all_ori_dataitems) {
        (this->all_X).push_back(Input(odi.x));
        (this->all_Y).push_back(this->getClassNo(odi.class_name));
    }
}

// read the file, split the dataset into test set 
// and training set, and initialize map_class_DS
bool DataTransformer::init(std::string dir, double split_rate) {
    this->fin.open(dir);
    if(!fin.is_open()) {
        return false;
    }

    int case_dim;
    fin >> case_dim;
    while(!fin.eof()) {
        OriDataItem ODI;
        for(int i = 0; i < case_dim; i++) {
            double temp;
            fin >> temp;       
            ODI.x.push_back(temp);
        }
        fin >> ODI.class_name;
        (this->all_ori_dataitems).push_back(ODI);

        (this->all_classes).insert(ODI.class_name);
    }

    fin.close();

    // number every class, and establish bi-directional map between number and class
    int class_index = 1;
    for(const auto& name : this->all_classes) {
        map_class_no[name] = class_index;
        map_no_class[class_index] = name;
        class_index++;
    }

    this->shuffle(); // shuffle the items that were read in just now
    this->transformDataitems(); // transform original dataitems into acceptable X and Y

    Dataset DS;
    DS.init(this->all_X, this->all_Y); 
    // notice that y's value of every item in DS is among {1.0, 2.0, 3.0}
    // which refers to three classes respectively
    const int DS_size = DS.getSize();
    this->train_set = DS.split(0, DS_size*split_rate-1);
    this->test_set = DS.split(DS_size*split_rate, DS_size-1);

    for(const auto& class_name : all_classes) {
        (this->map_class_DS)[class_name] = (this->train_set)->modify(this->getClassNo(class_name));
    }

    return true;
}

std::string DataTransformer::getClassName(int no) {
    return this->map_no_class[no];
}

int DataTransformer::getClassNo(std::string class_name) {
    return this->map_class_no[class_name];
}

std::vector<std::string> DataTransformer::getAllClasses() {
    std::vector<std::string> class_vec;

    for(const auto& class_name : this->all_classes) {
        class_vec.push_back(class_name);
    }

    return std::move(class_vec);
}

std::vector<std::unique_ptr<Dataset> > DataTransformer::getAllDS() {
    std::vector<std::string> class_vec;
    for(const auto& class_name : this->all_classes) {
        class_vec.push_back(class_name);
    }

    std::vector<std::unique_ptr<Dataset> > pDS_vec;
    for(const auto& class_name : class_vec) {
        pDS_vec.push_back((this->map_class_DS)[class_name]->duplicate());
    }

    return pDS_vec;
}

TEST_SET_TYPE DataTransformer::getTestSet() {
    auto X = this->test_set->getX();
    auto Y = this->test_set->getY();
    std::vector<std::string> modified_Y;

    for(const auto& y : Y) {
        modified_Y.push_back(this->getClassName(y));
    }

    return std::make_tuple(X, modified_Y);
}

int DataTransformer::getTestSetSize() {
    return (this->test_set)->getSize();
}

int DataTransformer::getTrainSetSize() {
    return (this->train_set)->getSize();
}