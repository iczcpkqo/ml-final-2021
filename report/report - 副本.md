# Final Assignment 2021

> Student Name: Xiang Mao
> Student Number: 21332237

## Q1

### What is the chosen dataset?

The stations in the city centre are those located near the median latitude of all stations, and the stations on the outskirts of the city are those with the least longitude of all stations. The stations are as follows:

1. Station: Exchequer Street
   * Number: 9
   * Latitude: 53.343033
   * Longitude: -6.263578
   * Location: In the city centre

2. Station: Kilmainham Gaol
   * Number: 97
   * Latitude: 53.342113
   * Longitude: -6.310015
   * Location: In the suburbs

I chose these two stations for the following reasons:

1. the two stations need to be located far enough away to have the most different behavioural data possible;
2. the station on the outskirts of the city is located at the lowest longitude, thus ensuring that it is far enough away from the city centre;
3. the coordinates of the station in the city centre are around the median latitude of all stations, so that this station reflects the behavioural data of the city centre station as much as possible.

### Feature engineering

* The raw data has a total of 11 features, which are related as follows.
  1. 'NAME', 'ADDRESS' and 'STATION ID' correspond one-to-one;
  2. $BIKE STANDS = AVAILABLE BIKE STANDS + AVAILABLE BIKES$
  3. for 'STATUS', where the data for `STATUS=open` is available. 4.
  4. For the same station, the values of 'LATITUDE' and 'LONGITUDE' are fixed, so this feature is an invalid feature that does not affect the predicted occupancy, so delete these two features from the forecast.

* Construct new features as follows:
$$
STAND\_OCCUPANCY = \frac{(BIKE STANDS-AVAILABLE BIKES)}{BIKE STANDS}
$$

* In summary, the features used are: 'STATION ID', 'TIME', 'BIKE STANDS', 'AVAILABLE BIKE', 'STANDS', 'STAND_OCCUPANCY'. 

* The labels used are as follows:
  1. 'STAND_OCCUPANCY_10': occupancy after 10 minutes of the corresponding time. 2. 'STAND_OCCUPANCY_10': occupancy after 10 minutes of the corresponding time.
  2. 'STAND_OCCUPANCY_30': Occupancy after 30 minutes of the corresponding time. 3.
  3. 'STAND_OCCUPANCY_60': Occupancy after 60 minutes of the corresponding time.

* The following graph shows the heat map between 'STATION ID', 'AVAILABLE BIKE STANDS' and 'STAND_OCCUPANCY' and label('STAND_OCCUPANCY_10'):

    ==Figure==

The magnitude of the correlation coefficient shows that 'STAND_OCCUPANCY' is highly correlated with label and is a very important feature, and 'STAND_OCCUPANCY' has a correlation coefficient of 0.85 with 'AVAILABLE BIKE STANDS', which also has a high correlation.



### LSTM method

I am using two machine learning methods: LSTM and Random Forest.

LSTM (Long short-term memory) is a special type of RNN that can be used to solve the gradient explosion and gradient disappearance problems that occur during training of long sequences. The LSTM is able to obtain better results on longer sequences than a normal RNN.

The loss function uses MSE and the optimizer is as follows:

```python
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
```

The above optimizer uses the ``Adam(Adaptive Moment Estimation)`` algorithm. Adam(Adaptive Moment Estimation) is essentially RMSprop with momentum terms, which dynamically adjusts the learning rate of each parameter using first-order moment estimation and second-order moment estimation of the gradient. The main advantage is that, after bias correction, the learning rate for each iteration has a defined range, making the parameters relatively smooth.

* The first parameter is: the parameter that can be used for iterative optimisation or the dicts that define the parameter set.
* The second parameter, lr, is: the learning rate (default: 0.001)

* LSTM parameters
  1. input_size: the number of corresponding features. 2. output_size: the number of predictions.
  2. output_size: the number of predictor variables and the number of data labels.
  3. hidden_layer_size: the number of features in the hidden layer, i.e. the number of neurons in the hidden layer.
  4. BATCH_SIZE: The size of this parameter will determine the number of samples to be trained at a time. It will affect how fast and how well the model can be optimized.

```python
  nn.LSTM(input_size,hidden_layer_size)
  nn.Linear(hidden_layer_size,output_size)
```

#### 97 stations

Baseline for a model that predicts the average of occupancy rates each time.

##### 10-minute predicted random comparison

  == 10-minute predicted random comparison ==

Figure 1.2.1 is the 'Random comparison of 10 minute forecast results'.

The results in the above figure show that the predictions for the next 10 minutes are highly accurate, so it is feasible to use the LSTM method to predict the occupancy rate for the next 10 minutes.

### Random forest method

The MSE is used as the evaluation metric for the random forest prediction, and the following parameters are selected from the random forest:

```python
  bootstrap=True, criterion='mse', max_depth=None,
  max_features='auto', max_leaf_nodes=None,
  min_impurity_decrease=0.0,
  min_impurity_split=None,
  min_samples_leaf=1, min_samples_split=4,
  min_weight_fraction_leaf=0.0,
  n_estimators=200, n_jobs=1,
  oob_score=False, random_state=None,
  verbose=0, warm_start=False
```

  1. bootstrap: Whether bootstrap sampling is used in the tree creation process. 2.
  2. criterion: A measure of branch quality.
  3. max_depth: the maximum depth of the tree. If `None`, the tree will continue to grow until the impurity of all leaf nodes is 0, or until each leaf node contains fewer samples than the input in the parameter `min_samples_split`.
  4. min_samples_split: The minimum number of samples needed for an intermediate node to branch. If a node contains fewer samples than the number entered in min_samples_split, the node will not branch.

#### 97 stations

##### 10 minutes Random forest parameters

#### 10, 30 ,60, random comparison

#### 10, 30 ,60, continuous comparison

Figure 1.3.1: 10-minute random forest parameters
Figure 1.3.2: 10, 30 ,60, randomized comparison
Figure 1.3.3: 10, 30 ,60, continuous comparison

==97 random forest plot==

The graph above shows the results of 'Station: Kilmainham Gaol' using Random Forest for the next 10, 30 and 60 minutes. The comparison between the results in the above graph and the baseline shows that the predictions are significantly better than the baseline results, and therefore the predictions for the next 10, 30 and 60 minutes are feasible using the Random Forest method.

#### 9 stations

#### 10, 30 ,60, random comparison

#### 10, 30 ,60, continuous comparison

#### Comparison of random forest effects for different stations

==9 random forest diagram==

The graph above shows the predictions for the next 10, 30 and 60 minutes for 'Station: Exchequer Street' when using Random Forest. The comparison between the above results and the baseline shows that the prediction results are significantly better than the baseline results, so the prediction of the next 10, 30 and 60 minutes using the random forest method is feasible.

* The prediction of 10, 30 and 60 minutes ahead for stations with different behavioral data by LSTM and Random Forest is significantly better than the baseline with the mean value of each prediction, so the prediction by LSTM and Random Forest is feasible.

## Q2

### What is a ROC curve. How can it be used to evaluate the performance of a classifier.
> What is a ROC curve. How can it be used to evaluate the performance of a classifier.

1. ROC stands for Receiver Operating Characteristic. The ROC takes the form of a curve drawn in the Cartesian coordinate system, which is called a ROC curve. The horizontal coordinate is the false positive rate (FPR) and the vertical coordinate is the true positive rate (TPR).
$$
    TPR = TP/(TP+FN) \\\
    FPR = FP/(FP+TN)
$$
TP means that the prediction is true, but it is also true; TN means that the prediction is false, but it is false; FP means that the prediction is true, but it is false; FN means that the prediction is false, but it is true.
The AUC (Area Under roc Curve) can be used to measure how good the ROC curve is. The value of AUC indicates the size of the area under the ROC curve on the horizontal axis.

### Give two examples of situations when a linear regression would give inaccurate predictions. Explain your reasoning.
> Give two examples of situations when a linear regression would give inaccurate predictions. Explain your reasoning.

The problem of inaccurate predictions can arise when the data distribution does not meet the assumptions of linear regression.

 1. Linearity
When using a linear regression model, it is expected that there is a linear relationship between the predictor variable x and the response variable y in the data. It is then reasonable to interpret the data by means of a straight line when x is only one dimensional, or by means of a surface when x is high dimensional. In the next two plots, there is clearly a linear relationship in Figure 1, but not in Figure 2. So it is possible that figure 1 can be used with linear regression, whereas figure B should not be used directly with linear regression.



![image-20220104233135580](D:/sync/dropbox/im/notes/data/resource/image/image-20220104233135580.png)

![image-20220104233112405](D:/sync/dropbox/im/notes/data/resource/image/image-20220104233112405.png)

2. Equivariance

Suppose we want to examine how household income affects the amount of luxury goods consumed, where household income is the predictor variable and the amount of luxury goods consumed is the response variable.
For poor households, most of them cannot afford to buy luxury goods, so the amount spent on luxury goods is very small for poor households, which results in a very small variance.
For exceptionally wealthy households, the purchase of luxury goods is not a problem at all, so the purchase of luxury goods is simply a matter of personal preference for each household, and the variance of luxury goods consumption is very large.

The above data clearly violate the assumption of homoscedasticity.
The results of a linear regression are determined by each variable and its weight, with the larger the weight, the larger the effect on the results. Therefore, the results are largely determined by a few highly weighted variables. This creates a model that not only loses information from the original data.

### Discuss three pros/cons of an SVM classiﬁer vs a neural net classiﬁer.
> 

1. neural networks

 * Advantages.
    a) The learning rules are simple and easy to implement by computer.
    b) Neural networks have strong non-linear fitting ability and can map arbitrarily complex non-linear relations.
    c) It has strong memory, powerful self-learning ability and non-linear mapping ability, so it has a wide range of applications.

  * Disadvantages.
    a) Difficult to explain the reasoning process and the basis of reasoning
    b) When there is not enough data, the neural network cannot work.
    c) Since all problems and features are converted into numbers, all inference is numerically computed, so some information is lost.

2. Advantages and disadvantages of SVM

Advantages.
(1) Non-linear mapping is the theoretical basis of the SVM method, which uses inner product kernel functions instead of non-linear mapping to higher dimensional spaces.
(2) The optimal hyperplane for feature space partitioning is the goal of SVM, and the idea of maximizing classification margins is the core of the SVM method.
(3) Support vector is the training result of SVM, and it is the support vector that plays a decisive role in SVM classification decision.
(4) SVM is a novel small sample learning method with a solid theoretical foundation. It does not involve probability measures or the law of large numbers, so it is different from existing statistical methods. In essence, it avoids the traditional process of induction to deduction and enables efficient "transductive reasoning" from training samples to forecast samples, greatly simplifying the usual problems of classification and regression.
(5) The final decision function of SVM is determined by only a small number of support vectors, and the complexity of the computation depends on the number of support vectors rather than the dimensionality of the sample space, which in a sense avoids the "dimensionality disaster".
(6) A small number of support vectors determines the final result, which not only helps us to catch the key samples and "eliminate" a large number of redundant samples, but also predestines the method to be not only simple but also "robust". This "robustness" is mainly reflected in:
(i) the addition or deletion of non-support vector samples has no effect on the model;
(ii) The support vector sample set has some robustness;
(iii) In some successful applications, the SVM method is not sensitive to the selection of kernels.
Disadvantages.
(1) SVM algorithm is difficult to implement for large training samples
Since SVM solves support vectors by means of quadratic programming, solving quadratic programming will involve the computation of a matrix of order m (m is the number of samples), and the storage and computation of this matrix will consume a lot of machine memory and computing time when the number of m is large.
(2) Difficulties in solving multi-classification problems with SVMs
The classical support vector machine algorithm only gives a two-class classification algorithm, but in the practical application of data mining, it is usually necessary to solve the classification problem of multiple classes. The problem can be solved by combining several two-class support vector machines. The main methods are one-to-many combinatorial models, one-to-one combinatorial models, and SVM decision trees; and by constructing combinations of multiple classifiers. The main principle is to overcome the inherent shortcomings of SVM and combine the advantages of other algorithms to solve the classification accuracy of multi-class problems. For example, it can be combined with coarse-set theory to form a combination of classifiers for multi-class problems with complementary advantages.


### Describe the operation of a convolutional layer in a convNet. Give a small example to illustrate.
> 

### In k-fold cross-validation a dataset is resampled multiple times. What is the idea behind this resampling i.e. why does resampling allow us to Give a small example to illustrate.
> 









