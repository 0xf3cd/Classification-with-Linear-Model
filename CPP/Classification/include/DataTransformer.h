// this class intends at reading the file where original data is saved,
// and then transforming the data into another format which is acceptable 
// for the class IrisClassfier.

#include "Dataset.h"
#include <fstream>
#include <set>
#include <vector>
#include <map>
#include <tuple>
#include <string>

// typedef std::unordered_map<std::string, LinearRegression> MyMap;
typedef std::tuple<std::vector<Input>, std::vector<std::string> > TEST_SET_TYPE;

class DataTransformer {
private: 
    std::ifstream fin;
    std::vector<OriDataItem> all_ori_dataitems;
    std::vector<Input> all_X;
    std::vector<double> all_Y;
    std::set<std::string> all_classes;
    std::map<std::string, int> map_class_no;
    std::map<int, std::string> map_no_class;
    std::unique_ptr<Dataset> test_set;
    std::unique_ptr<Dataset> train_set;
    std::map<std::string, std::unique_ptr<Dataset> > map_class_DS;

    // shuffle all_ori_dataitems
    void shuffle();

    void transformDataitems();

public: 
    // read the file, split the dataset into test set 
    // and training set, and initialize map_class_DS.
    bool init(std::string dir, double split_rate=0.9);

    std::string getClassName(int no);
    int getClassNo(std::string class_name);

    std::vector<std::string> getAllClasses();
    std::vector<std::unique_ptr<Dataset> > getAllDS();

    TEST_SET_TYPE getTestSet();

    int getTestSetSize();
    int getTrainSetSize();
};