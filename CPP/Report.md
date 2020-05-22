# 机器学习大作业 #1



## 一、实验概述

大作业 1 要求选用不同的线性学习方法分别实现一个分类任务、一个回归任务。注意不可以使用库和框架。



作业主要实现了线性回归和逻辑斯特回归两个模型，并使用这两个模型解决了一个回归任务和两个分类任务（基于鸢尾花和心脏病数据集）。



其中回归任务是通过随机生成一个线性函数 `f(X) = W^T·X+b` ，其中参数矩阵 `W` 及偏置 `b` 都是随机生成的，并根据 `f(X)` 随机生成若干数据用于回归模型的训练。在线性回归模型训练完毕后，可以将模型的训练结果（训练得到的各参数）和之前随机生成的参数进行比较。



分类任务中使用了两个数据集。一个是入门级、非常常见的鸢尾花数据集 `Iris`，每个数据有 4 个维度，共有 3 类；另外一个则是来源自 kaggle 的心脏病数据集，每个数据有 13 个维度，共分为两类（有/无心脏病）。





## 二、实验方案设计

以下内容以 `Classification` 文件夹中的内容为例来说明。



### 2.1 总体设计思路和总体架构

```
                       <--  LinearRegression  <--
                      /								           \
main.cpp  <--  Classifier                       Dataset <-- DataTransformer
                      \ 					               /
                       <-- LogisticRegression <--

```



数据集储存在文件中，类 `DataTransformer` 负责从文件中读取数据，并生成类 `Dataset` 的对象。



在将数据集的内容读取到内存中后（即 `Dataset` 对象中），需要将数据集划分为训练集和测试集，并交由模型（`LinearRegression` 或 `LogisticRegression` 进行训练）。至此，已经可以解决回归问题了。



如果需要使用回归模型解决分类问题，则需要使用类 `Classifier`。`Classifier` 中为分类问题的每一个类都生成了对应的回归模型进行预测，如某个分类问题中涉及到三个类，则 `Classifier` 中会储存三个`LogisticRegression` 对象，由这些对象共同完成分类操作。由于采用的是 One-VS-One 的方法，所以每一次预测时，分类器都会返回一个向量，其中每一项代表这一个 case 属于某一类的概率。如 `[0.9, 0.11, 0.03]`，这就表示分类器预测这一个 case 是属于第一类的。case 属于某一类的概率是由单独的线性回归或者逻辑回归模型计算的，所以有多少个类就需要多少个回归模型。



最终在 `main.cpp` 中将所有代码拼接起来，并输出训练和分类的各种结果。



### 2.2 算法核心和基本原理

#### 2.2.1 Class  Dataset 定义

```C++
typedef std::vector<double> Input;

struct OriDataItem {
  Input x;
  std::string class_name;
};

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

  // return a new dataset with the same contents
  // if y's value of a item equals to class_no (the item belongs 
  // to the class), then the y's value will be modified to 1.0, with
  // other items' being 0.0
  std::unique_ptr<Dataset> modify(double class_no);
};
```



`Dataset` 的定义比较简明。对于一个 `Dataset` 对象，首先使用 `init` 函数传入需要它保存的所有数据项（其中 x 存放在 `vector<Input>` 中，`Input` 本质为 `vector<double>`，即 x 的每一维的值都被储存在一个 vector 中；y 的值被存放在 `vector<double>` 中）。



值得一提的是，可以调用 `Dataset` 对象的 `duplicate` 函数来赋值一个新的数据集，或是通过 `split` 函数来获取数据集的一个子集。这两个函数的返回值都是智能指针类型的，使用时需要注意及时 `move` 所有权。



#### 2.2.2 Class DataTransformer 定义

```C++
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

  // perform shuffle on all_ori_dataitems
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
```



简单来说，通过 `init` 进行初始化。其中数据集所在的文件路径为 `dir`，训练集和测试集的划分比例为 `split_rate`，默认为 0.9。



在读入文件后，需要对读入的数据项进行打乱操作，具体是由 `shuffle` 函数完成的。

```C++
// shuffle all_ori_dataitems
void DataTransformer::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());

    auto& items = this->all_ori_dataitems;
    std::shuffle(items.begin(), items.end(), g);
}
```



通过 `getAllClasses` 和 `getAllDS` 可以得到数据集文件中的所有类以及对于每一个类进行调整的数据集。由于解决分类问题时使用的是 One-VS-One 的方法，所以训练每一个回归模型前都需要对数据集进行调整，即将从属于这一类的 cases 的 y 值改为 1，其他的 cases 的 y 值改为0。



#### 2.2.3 Class LinearRegression 定义

```C++
typedef std::tuple<int, double, double> TrainRet;

class LinearRegression {
private:
  std::unique_ptr<Dataset> pDS;
  std::vector<double> coef; // coef for "coefficient"
  double bias;
  double alpha; 
  int iter; // how many iterations the training process will perform
  double cost_threshold; // when the cost value reaches the threshold, stop iteration

  // calculate the cost value (namely the value of function J)
  double calcCostValue(std::vector<double> predicted_Y);

public:
  LinearRegression(std::unique_ptr<Dataset> pDS);
  void setAlpha(double alpha);
  void setIter(int iter);
  TrainRet train();

  // perform prediction on one item
  double predictOnce(Input x);

  // use the trained model to perform prediction on 
  // items in X one by one
  std::vector<double> predict(std::vector<Input> X);

  // return the coefficients
  std::vector<double> getCoef();

  double getBias();
};
```



`LinearRegression` 类只能使用有参构造函数，且需要传入一个智能指针，它指向一个数据集对象（训练集）。初始化时，通过 `#include <random>` 引入随机数相关的 API，来随机对所有参数和偏置赋随机初值。



之后可以通过 `train` 函数来对模型进行训练，它将返回一个 `tuple`，其中包含了 `iteration` 次数，`cost value` 的变化等。



`cost value` 计算函数如下：

```c++
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
```



训练过程对应的代码如下：

```C++
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
```



关键部分我已经添加了注释，所以在此不再赘述。模型训练完毕后，可以通过 `predict` 或者 `predictOnce` 来对一组数据或者单个数据进行预测。



#### 2.2.4 Class LogisticRegression 定义

Logistic 回归模型和线性回归模型非常相似，其类定义中多了 `sigmoid` 函数，如下。

```c++
double LogisticRegression::sigmoid(double Z) {
  return 1 / (1 + exp(-Z));
}
```



`cost value` 的计算过程相应变更如下：

```c++
// calculate the cost value (namely the value of function J)
double LogisticRegression::calcCostValue(std::vector<double> predicted_Y) {
  const int DS_size = (*this->pDS).getSize();
  std::vector<double> Y = (*this->pDS).getY();

  if(Y.size() != predicted_Y.size()) { // mismatch
    return -1.0;
  }

  double cost_value = 0.0;
  for(int i = 0; i < Y.size(); i++) {
    cost_value += Y[i] * log(predicted_Y[i]) + (1 - Y[i]) * log(1 - predicted_Y[i]);
  }
  cost_value *= -1.0 / double(DS_size);

  return cost_value;
}
```



训练（递归下降）和预测过程和线性回归模型类似，在此不赘述了。



#### 2.2.5 Class Classifier 定义

```c++
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
```



首先，通过 `#ifdef USE_LINEAR` 来决定分类器是使用 `LinearRegression` 还是 `LogisticRegression`（当然默认情况下应当使用 Logistic 回归模型）。在构造对象时，需要传入所有类的类名（从而可以知道共有多少类），并传入每一个类对应的数据集。



之后通过调用 `train` 进行分类器的训练，实际上它内部对每个回归模型进行了单独的训练。

```c++
std::vector<TrainRet> Classifier::train() {
  std::vector<TrainRet> tr_vec;
  for(const auto& class_name : all_classes) {
    auto iter = map_class_LR.find(class_name);
    auto train_detail = (iter->second).train();
    tr_vec.push_back(train_detail);
  }

  return std::move(tr_vec);
}
```



与 `LinearRegression` 及 `LogisticRegression` 类似，`Classifier` 也提供了多个函数进行分类预测。值得一提的是，带有 `WithDetail` 的函数将返回每一个回归模型的预测结果（类似于返回一个 `[0.03, 0.98, 0.22]` 的向量），不带 `WithDetail` 的函数则直接返回分类结果（字符串）。



### 2.3 项目组织

代码文件夹根目录下文件分布如下：

```
.
├── Classification
└── Regression
```



 `Regression Folder` 

```
.
├── CMakeLists.txt
├── build
├── include
└── src

3 directories, 1 file
```

其中 `Regression` 中解决的是一个回归任务，其中使用随机生成的数据集训练线性回归模型，其中内容如下。



其中头文件放在 `include` 中，其中还有用于格式化输出的开源库 `tabulate`。`.cpp` 文件存放在 `src` 中。如果需要编译和运行代码，则可以进入 `build` 文件夹，而后执行 `cmake ..` 和 `make` 即可完成编译操作。



`Classification Folder`

```
.
├── CMakeLists.txt
├── build
│   ├── CMakeCache.txt
│   ├── CMakeFiles
│   ├── Makefile
│   ├── cmake_install.cmake
│   └── Classification
├── dataset
│   ├── heart
│   ├── heart.csv
│   ├── iris
│   └── iris.csv
├── include
│   ├── Classifier.h
│   ├── DataTransformer.h
│   ├── Dataset.h
│   ├── Input.h
│   ├── LinearRegression.h
│   ├── LogisticRegression.h
│   └── tabulate
└── src
    ├── Classifier.cpp
    ├── DataTransformer.cpp
    ├── Dataset.cpp
    ├── LinearRegression.cpp
    ├── LogisticRegression.cpp
    └── main.cpp

6 directories, 21 files
```

`Classification` 解决的是分类问题。共有两个数据集供使用，都存放在 `dataset` 文件夹中，其中 `heart` 指的是心脏病数据集，而 `iris` 则是鸢尾花数据集。默认使用数据集 `heart` ，如果需要使用鸢尾花数据集，则需要在编译时指定（见下）。



```cmake
CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

PROJECT(Classification)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED True)

INCLUDE_DIRECTORIES(include)

AUX_SOURCE_DIRECTORY(src DIR_SRC)
ADD_EXECUTABLE(Classification ${DIR_SRC})

# ADD_DEFINITIONS(-D DATASET_IRIS)
```



如果取消最后的注释，则会在编译时传入宏定义，从而使用  `Iris` 数据集。





## 三、 实验过程

### 3.1 环境说明

实验运行在 macOS 操作系统上，使用的语言是 C++，**编译器必须支持 C++17 ，否则不能通过编译**（使用了 C++17 的一些语法和标准库）。

项目使用了 cmake 进行编译管理，后续在 macOS 上需要使用 make 工具，编译器使用的是 clang++。

项目使用了开源库 `tabulate` 进行格式化输出，并使用了 `set, map, vector...` 等 C++ STL。



### 3.2 源代码文件清单、主要函数清单

主要函数见上一部分。



源代码文件结构如下：

```
.
├── Classification
│   ├── CMakeLists.txt
│   ├── build
│   │   ├── CMakeCache.txt
│   │   ├── CMakeFiles
│   │   ├── Makefile
│   │   ├── cmake_install.cmake
│   │   └── Classification
│   ├── dataset
│   │   ├── heart
│   │   ├── heart.csv
│   │   ├── iris
│   │   └── iris.csv
│   ├── include
│   │   ├── Classifier.h
│   │   ├── DataTransformer.h
│   │   ├── Dataset.h
│   │   ├── Input.h
│   │   ├── LinearRegression.h
│   │   ├── LogisticRegression.h
│   │   └── tabulate
│   └── src
│       ├── Classifier.cpp
│       ├── DataTransformer.cpp
│       ├── Dataset.cpp
│       ├── LinearRegression.cpp
│       ├── LogisticRegression.cpp
│       └── main.cpp
└── Regression
    ├── CMakeLists.txt
    ├── build
    │   ├── CMakeCache.txt
    │   ├── CMakeFiles
    │   ├── Makefile
    │   ├── cmake_install.cmake
    │   └── Regression
    ├── include
    │   ├── Dataset.h
    │   ├── Input.h
    │   ├── LinearRegression.h
    │   └── tabulate
    └── src
        ├── Dataset.cpp
        ├── LinearRegression.cpp
        └── main.cpp
```

 

### 3.3 实验结果展示

#### 3.3.1 Regression

在进入 `Regression` 目录后，创建 `build` 文件夹（如果不存在）并进入。之后执行 `cmake ..`。

![image-20200517224349476](/Users/Jake/Library/Application Support/typora-user-images/image-20200517224349476.png )



接下来执行 `make` 进行编译操作。

![image-20200517224434985](/Users/Jake/Library/Application Support/typora-user-images/image-20200517224434985.png)



编译成功后出现可执行文件 `Regression`，执行它。

程序运行时，首先输出了数据集的一些信息。并在训练结束后输出了迭代次数、`cost value` 等信息。

![image-20200517224554124](/Users/Jake/Library/Application Support/typora-user-images/image-20200517224554124.png)



随后输出的是各个参数的真实值，以及模型学习到的值。

![image-20200517224847851](/Users/Jake/Library/Application Support/typora-user-images/image-20200517224847851.png)



最后输出了模型在测试集上的运行结果。当真实值和估计值相差 `0.02` 以上的时候，将被标红。

![image-20200517225004854](/Users/Jake/Library/Application Support/typora-user-images/image-20200517225004854.png)



#### 3.3.2 Classification

首先使用心脏病数据集（默认情况）。与之前类似，进入 `build` 目录后（没有的话需要创建），使用 cmake 和 make 进行编译。

![image-20200517225236133](/Users/Jake/Library/Application Support/typora-user-images/image-20200517225236133.png)



之后目录下出现可执行文件 `Classification`。

执行后首先输出数据集的情况，随后输出子分类器的迭代和代价值等情况。

![image-20200517230404306](/Users/Jake/Library/Application Support/typora-user-images/image-20200517230404306.png)



之后将输出在测试集上的分类情况，最后一列标志着分类是否正确。

![image-20200517230434094](/Users/Jake/Library/Application Support/typora-user-images/image-20200517230434094.png)



接下来换成使用鸢尾花数据集。在对应的 `CMakeLists.txt` 中添加宏定义，如下：

```cmake
ADD_DEFINITIONS(-D DATASET_IRIS)
```



之后重新生成项目并编译，并尝试运行。

![image-20200517231425521](/Users/Jake/Library/Application Support/typora-user-images/image-20200517231425521.png)

![image-20200517231439900](/Users/Jake/Library/Application Support/typora-user-images/image-20200517231439900.png)



可以发现模型在鸢尾花数据集上有非常好的分类效果，在心脏病数据集上则有错分的情况。





## 四、总结

实验进展良好，通过手写各个回归模型和分类器，更深入地理解了各个算法的思想。并在过程中锻炼了动手能力、工程能力、团队合作能力等各个能力。



至于后续改进方向，可以考虑在数据预处理部分做一些工作，比如考虑数据归一化等等，同时可以对数据进行特征选择等操作。


