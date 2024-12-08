# UVA DS 7200: Distributed Computing Final Project
## Group Members: Clarence Williams, Lingzhen Zhu, Ethan Nelson

This is a repository for the UVA SDS Distributed Computing final project. The code for the project is split into multiple .py and .ipynb files. The table of contents of the files is listed below:

### Model Pipeline:

* [final_proj.ipynb](final_proj.ipynb): The first file we created to develop the model pipeline. Also includes the beginning code for logistic regression.
* [DataPipelineFxn.py](DataPipelineFxn.py): A python file that holds the code to put the data through the data pipeline, returning the training data with regular sampling, test data, and traing data with balance class distribution ("*GetSparkDF*"). Also has a function to return a spark data frame without putting it through the Pipeline (*GetBaseDataFrame*).
* [eda-notebook.ipynb](eda-notebook.ipynb): A notebook to explore the data before creating the model. Inlcudes several visualizations of the distirbuion of various features in the data set. This provided insight into what the data we were dealing with was like.

### Creating Models:

* [logistic_regression.ipynb](logistic_regression.ipynb): A notebook to develop the logistic regression model.
* [logistic_regression_2023_test_set.ipynb](logistic_regression_2023_test_set.ipynb): A notebook to develop the logistic regression model without class balancing and test on the 2023 Season.
* [random_forest.ipynb](random_forest.ipynb): A notebook to develop the random forest model without class balancing.
* [random_forest_undersmapling.ipynb](random_forest_undersmapling.ipynb): A notebook to develop the random forest model with class balancing.
* [random_forest_2023_test_set.ipynb](random_forest_2023_test_set.ipynb): A notebook to develop the random forest model without class balancing and test on the 2023 Season.
* [NaiveBayes.ipynb](NaiveBayes.ipynb): A notebook to develop the Naive Bayes model with and wihtout class balancing.
* [NaiveBayes_2023_test_set.ipynb](NaiveBayes_2023_test_set.ipynb): A notebook to develop the Naive Bayes model wihtout class balancing and test on the 2023 season.


### Miscelanous Files:

* [exploratory-data-analysis.ipynb](exploratory-data-analysis.ipynb): An empty file that was initially created to look at the data, but the purpose was replaced with [eda-notebook.ipynb](eda-notebook.ipynb).
* [final_proj_rf.ipynb](final_proj_rf.ipynb): An initial file to test the random forest, but this file is superceded by [random_forest.ipynb](random_forest.ipynb).
* [MLP.ipynb](MLP.ipynb): An attempt to create a Multilayer Perceptron model, that was not completed and can be ignored.
* [Untitled.ipynb](Untitled.ipynb): An empty file that can be ignored. 
