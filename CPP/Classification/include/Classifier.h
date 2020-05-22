#include "LinearRegression.h"
#include "LogisticRegression.h"
#include "Dataset.h"
#include <vector>
#include <map>
#include <unordered_map>
#include <string>

typedef std::map<std::string, double> Prediction;
#ifdef USE_LINEAR
    typedef LinearRegression ModelType;
#else
    typedef LogisticRegression ModelType;
#endif
typedef std::unordered_map<std::string, ModelType> MyMap;

class Classifier {
private:
    std::vector<std::string> all_classes;
    MyMap map_class_LR; // one class corresponds to one dataset

public: 
    // make sure that the names of the classes are different from each other!
    // class_names.size() should be equal to pDS_vector.size().
    // each class has its own database, and the items in a database are marked with
    // either "1" or "-1", where "1" means that the item belongs to the class
    explicit Classifier(std::vector<std::string> &class_names, std::vector<std::unique_ptr<Dataset> > &pDS_vector);

    std::vector<TrainRet> train();

    // perform prediction on one item
    // return the class that x is most likely to belong to
    std::string predictOnce(Input x);

    // perform prediction on one item
    // return the probabilities that the item belongs to every class
    Prediction predictOnceWithDetail(Input x);

    // perform predictOnce on each item in X
    std::vector<std::string> predict(std::vector<Input> X);

    // perform predictOnceWithDetail on each item in X 
    std::vector<Prediction> predictWithDetail(std::vector<Input> X);
};