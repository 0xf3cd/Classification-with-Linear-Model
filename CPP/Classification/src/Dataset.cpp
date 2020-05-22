#include "Dataset.h"

/*
class Dataset {
private: 
    int size;
    std::vector<Input> X; // all the inputs
    std::vector<double> Y; // all the values
*/


// return boolean value to represent
// whether the initialization is successful
bool Dataset::init(std::vector<Input> X, std::vector<double> Y) {
    if(X.size() == Y.size()) { 
        this->size = X.size();
        this->input_size = X[0].size();
        this->X = std::vector<Input>(X);
        this->Y = std::vector<double>(Y);

        return true;
    } else {
        return false;
    }
}

// return the size of the dataset
int Dataset::getSize() {
    return this->size;
}

// return the size of the input(x)
int Dataset::getInputSize() {
    return this->input_size;
}

// return the index-th item in the dataset
// if the index is out of range, return nullptr
std::unique_ptr<DataItem> Dataset::getDataItem(int index) {
    if(0 <= index && index < this->size) {
        std::unique_ptr<DataItem> pDI(new DataItem);
        pDI->x = X[index];
        pDI->y = Y[index];

        return std::move(pDI);
    } else { // out of range
        return std::unique_ptr<DataItem>(nullptr);
    }
}

std::vector<Input> Dataset::getX() {
    return std::vector<Input>(this->X);
}

std::vector<double> Dataset::getY() {
    return std::vector<double>(this->Y);
}

// return a new dataset with the same contents
std::unique_ptr<Dataset> Dataset::duplicate() {
    std::unique_ptr<Dataset> pDS(new Dataset);

    pDS->size = this->size;
    pDS->input_size = this->input_size;
    pDS->X = std::vector<Input>(this->X);
    pDS->Y = std::vector<double>(this->Y);

    return std::move(pDS);
}

// return a subset of the dataset, namely Dataset[idx_s, idx_e]
std::unique_ptr<Dataset> Dataset::split(int idx_s, int idx_e) {
    if(idx_s < 0 || idx_e >= this->size || idx_e < idx_s) {
        return std::unique_ptr<Dataset>(nullptr);
    }

    std::unique_ptr<Dataset> pDS(new Dataset);

    pDS->size = idx_e - idx_s + 1;
    pDS->input_size = this->input_size;
    pDS->X = std::vector<Input>((this->X).begin()+idx_s, (this->X).begin()+idx_e+1);
    pDS->Y = std::vector<double>((this->Y).begin()+idx_s, (this->Y).begin()+idx_e+1);

    return std::move(pDS);
}

// return a new dataset with the same contents
// if y's value of a item equals to class_no (the item belongs 
// to the class), then the y's value will be modified to 1.0, with
// other items' being 0.0
std::unique_ptr<Dataset> Dataset::modify(double class_no) {
    std::unique_ptr<Dataset> pDS(new Dataset);

    pDS->size = this->size;
    pDS->input_size = this->input_size;
    pDS->X = std::vector<Input>(this->X);
    // pDS->Y = std::vector<double>(this->Y);
    for(const auto& y : this->Y) {
        (pDS->Y).push_back(class_no==y ? 1.0 : 0.0);
    }

    return std::move(pDS);
}