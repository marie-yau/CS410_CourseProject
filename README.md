# CourseProject

## Presentation Link
https://youtu.be/pO3tHq0Es44

## Overview of the Function of the Code
This project introduces a library dedicated to optimizing parameters for various MeTA estimators. This library is tailored to facilitate parameter tuning across multiple MeTA estimator types, such as OkapiBM25, PivotedLength, AbsoluteDiscount, JelinekMercer, and DirichletPrior. An example of how the parameters can be specified for each MeTA estimator can be seen in the ESTIMATOR_TO_PARAM_GRID dictionary in main.py.

This project adopts an unconventional strategy for parameter tuning by capitalizing on the grid search and random search functionalities within the scikit-learn library. While these tuning methods in scikit-learn are conventionally designed for hyperparameter optimization in machine learning models, applying the same techniques to MeTA ranking functions necessitate the development of this innovative approach.

An example usage of this library can be seen in the 'if __name__ == '__main__':' block in main.py, wherein a configuration file is read, processed, and a dataset is partitioned into training and testing sets. Subsequently, this example systematically performs parameter tuning for each MeTA estimator using both grid and random search methods, capturing the optimal model and its corresponding test score at each iteration.

Upon completion of the parameter tuning process, we generate a visual representation of the test scores for each iteration, saving the resulting plot as a PNG image, while outputting the MeTA model with the highest test score to the console.

## Software Implementation

* Configuration Processing

The application reads the provided configuration file (`config.toml`). Configurations include query-runner configurations (query path and start-id) and Estimator searching configurations.

* Data Splitting

The application splits the data into training and testing sets based on the split_ratio specified in the configuration file. This is crucial for evaluating the performance of the models on unseen data.

* Model Evaluation and Selection

The application iterates over a predefined set of estimators and their respective parameter grids to perform a search (e.g., grid search or random search) to find the best model based on the training data. The performance of these models is then assessed using the test data.

* Parameter Tuning Library Design

In order to use GridSearchCV and RandomizedSearchCV from the sklearn.model_selection library, each pair of MeTA ranker (e.g. Okapi BM25) and scoring function (e.g. NDCG) needs to be encapsulated within a scikit-learn Estimator. In order to facilitate code-reuse, we first implemented a derived class of sklearn.base.BaseEstimator that encapsulates the scoring function (see ndcg_estimator.py). Afterwards, we create derived classes of the NdcgEstimator class for each MeTA ranker (e.g. OkapiBM25NdcgEstimator). This way, each MeTA ranker Estimator subclass only needs to set the MeTA ranker, while the core scoring functionality that is common is implemented in the NdcgEstimator class. This pattern can be extended for scoring functions other than NDCG, through an enhancement to the library.

* Parameter Tuning Implementation

Given that we created Estimators for each MeTA ranker, scoring function pair, performing grid search and random search is a straightforward call to first creating the corresponding GridSearchCV and RandomizedSearchCV objects, with a subsequent call to the .fit() method.

* Result Analysis and Visualization

After evaluating all models, the application identifies and displays the overall best model and its test score. It also generates a bar plot showing the test scores for each model iteration.

* Clean Datasets

We used three datasets - Cranfield, CISI, and NPL. While the Cranfield dataset was already in the required format, the CISI and NPL datasets were not. Therefore, we had to clean the datasets to make them adhere to the required format. The required format:

document files - each document must be on one line

query file - each query must be on one line

relevance file - each line must contain query number, document number, and relevance score separated by one space

Code for cleaning the datasets can be found in src/meta/clean_datasets folder.

## Usage of the Software
* Create and activate a new Conda environment with Python 3.5 by running
`conda create -n myenv python=3.5`
`conda activate myenv`

* Install Required Packages: Install the necessary packages (pytoml, numpy, matplotlib, scipy) in your environment:
`conda install numpy matplotlib scipy`
`conda install -c conda-forge pytoml`

* Command line run of the main python application script
`pythonw src/main.py datasets/config.toml`

* Configure the data paths
Paths of queries and query judgment (relevances file) can be configured in config.tomls

* Software configuration
Estimator searching params can be configured in config.toml params-tuner configs, including
1. logging levels
2. split ratios for training/testing data
3. numfold of cross-validation

## Contribution of Each Team Member
*Marie*

Marie found and processed datasets that were needed for this project. We used three datasets - Cranfield, CISI, and NPL. While the Cranfield dataset was already in the required format, the CISI and NPL datasets were not. Therefore, I had to process the dataset to make them adhere to the required format. I authored all code in src/meta/clean_datasets folder that contains six python scripts. These scripts processes documents, queries, and relevance files for both CISI and NPL datasets.


*Daniel*

Daniel researched grid search and random search in scikit-learn, all parameters of the MeTA toolkit, and implemented parameter tuning to find the most optimal parameters based on evaluation functions like NDCG for each MeTA ranker. Despite the unconventional use of scikit-learn for hyperparameter tuning in the MeTA context, leveraging this standard machine learning library made the implementation less error-prone compared to developing grid search and random search algorithms from scratch.


*Xuan*


Xuan investigated and implement (1) validation and performance evaluation, including train test data splitting and model evaluation and recommendation (2) software integration, including software wrapper of the data processing, configuration process, parameter tuning, model integration training and testing layers (3) result analysis, metrics reporting and visualization.





