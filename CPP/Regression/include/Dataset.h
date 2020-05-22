#ifndef DATASET
    #define DATASET
    #include "Input.h"
    #include <vector>
    #include <memory>

    struct DataItem {
        Input x;
        double y;
    };

    class Dataset {
        private: 
            int size;
            int input_size;
            std::vector<Input> X; // all the inputs
            std::vector<double> Y; // all the values

        public:
            // return boolean value to represent
            // whether the initialization is successful
            bool init(std::vector<Input> X, std::vector<double> Y); 
            
            // return the size of the dataset
            int getSize(); 

            // return the size of the input(x)
            int getInputSize(); 
            
            // return the index-th item in the dataset
            // if the index is out of range, return nullptr
            std::unique_ptr<DataItem> getDataItem(int index); 

            std::vector<Input> getX();
            std::vector<double> getY();

            // return a new dataset with the same contents
            std::unique_ptr<Dataset> duplicate();

            // return a subset of the dataset, namely Dataset[idx_s, idx_e]
            std::unique_ptr<Dataset> split(int idx_s, int idx_e);
    };
#endif