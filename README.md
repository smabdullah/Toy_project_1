**Sound pressure level prediction**

This project works with the NASA dataset which contains various wind tunnel speeds and angles of attack. It predicts the sound pressure level.

This dataset is collected from https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise and known as Airfoil dataset. It has five feature attributes, all are numeric and one predicted output, the sound pressure level.

This is a regression problem. We start with the linear regression (OLS) and then try the decisiontree and random forest.

The accuracies of using different matrics are follows


**Model**          |    **R^2**     |         **MSE**      
-----------------------------------------------------------
Linear (OLS)       | 0.5585         |  20.7651             
Decision tree      | 0.8909         | 5.1321               
Random forest      | 0.9279         | 3.3899               

The current implementation does not consider k-fold cross validation and (hyper) parameter tunning. In the next run, I will use grid search to do cross validation and paramter tunning.

This particular dataset has five independent variables (feature matrix). We can use PCA (unsupervised) or LDA (supervised) to extract the most valuable features to visualise the regression outcome.


