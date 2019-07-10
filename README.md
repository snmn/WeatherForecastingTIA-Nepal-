# WeatherForecastingTIA-Nepal-
this repo deals with a project including various models trained with weather related data from TIA NEPAL and predicting the weather
1. Introduction 
Background 
Meteorology is the scientific study of the atmosphere that focuses on weather processes and forecasting. Weather forecasting is the application of current technology and science to predict the state of the atmosphere for a future time and a given location. Weather Forecasting is highly complex, multi-dimensional, dynamic and complicated process because it involves many entities of the atmosphere as well as many concepts of atmospheric physics. 
Weather Forecasting is one of the vital applications of Meteorology. Weather and Climate is a measure of time. A condition of the atmosphere for a short period of time is Weather, while, how the atmosphere behaves over a long period of time is Climate. Weather Nowcasting, is defined as a Short-term weather forecast, for the expected conditions in the next few hours. It is a description of the current weather parameters for the next 0 -2 hours and these days, Nowcasting, is getting popular. Out of the several meteorological elements the 5 major components, namely temperature, pressure, wind, humidity, and precipitation play a vital role in determining the weather and the climate of a place. Analysis of these meteorological elements forms the basis for forecasting the weather and to determine the climate of a location. 
 Forecasting & Prediction (Statistical Approach) The need for weather forecasts was felt from the early days of civilization. Later years of experience and scientific studies on the physics and dynamics of the atmosphere have resulted in the present state of art in weather forecasting. Conventional operational weather forecasting consists of four different steps: i. Maintaining a network of stations, recording surface and upper air observations. ii.  Collecting and exchanging the data with neighboring regions through a telecommunication system. iii.  Analyzing the data received and using them for preparing prognostic charts.  iv. Interpreting the prognostic charts in terms of weather parameters for different user interests  In meteorological parlance, weather forecasts are classified according to their range (period of validity).  Sort range forecasts have period of validity from a few hours up to three days. These are useful for aviation, planning day-to-day activities, sports, tourism and for taking precautionary measures against disastrous weather to save life and property.  
2  
Medium range forecasts have period of validity of three to ten days. They have considerable use in agricultural operations and planning, water resources management, flood warning, irrigation scheduling and strategic operation.  Long-range forecasts with validity of few weeks to a month or even a season are useful for planning purposes. Until recently, most of the work in the field of Long-Range Forecasting had a statistical or empirical approach. However, at present a lot of work is being done on the possible use of numerical models for long range forecasting.   
 Forecasting & Prediction (ANN approach) In recent years, a new mathematical tool in the form of Artificial Neural Networks (ANNs) has been available for performing nonlinear function approximation, pattern recognition, data compression, noise reduction etc. Although used for a wide variety of applications, the real power of ANNs, particularly that of the feed forward multilayered neural networks (FFNNs), lies in performing nonlinear function approximation and classification. The FFNNs have the ability to approximate any non-linear functional relationship between a set of input-output variables and it has triggered numerous applications including those in the non-linear identification and prediction of weather systems. It is to be noted that unlike the traditional linear or nonlinear regression techniques, the FFNNs do not require an explicit function al form for performing input- output mapping. In this sense they can be viewed as black-box models.  To impart into an FFNN, the above-mentioned capabilities, the neural network is made to learn the relationship between the input-output data procedure called network training. The set of such input-output patterns is known as the training set. The most popular FFNN training algorithm is the Error-Back-Propagation (EBP) proposed by Rumelhart et al and Werbos. Eisner and Tsonis have reviewed the applications of artificial neural networks to the prediction problems arising in meteorology and have given examples of time series forecasting.   
Objective and scope The objectives of this research study are 
• To examine the applicability of Regression Analysis approach by developing effective and efficient predictive models for weather analysis for Kathmandu Airport, Nepal. • To develop an efficient, reliable and effective weather forecasting system based on Regression Analysis. 
3  
The scope of this project is to organize the data in such a manner that the relationship becomes established between past conditions and the future behavior of Weather Parameters.  
Need and Social Significance of the project Among the natural disasters worldwide, 90% were weather related disasters, of which 45% was meteorological (storms), 33% were hydrological (floods), 12% were climatological (heat waves, cold waves, droughts, wildfires) and 10% were geophysical events (earthquakes and volcanic eruptions). Losses and damages due to severe Rainfall and Storm are enormous, which causes irreparable losses to mankind. So, this project work concentrates on prediction of weather parameters (Mean Temperature) and similarly can also be used to predict rainfall, storm etc. 
An effective disaster prevention model and weather alerts developed may avoid heavy losses that are incurred by weather related disaster. Also, a reliable prediction model will serve different sectors like agriculture, common man and other meteorological dependent industries.  
4  
2. Methodology 
Regression Model has been used for the purpose of prediction and forecasting the different meteorological parameters like Temperature, Relative Humidity and Rainfall. One of the method for this is the Linear regression model. Prediction essentially involves determining the functional relationship between the input and output variables of a system. Once such relationship is established from the available input output data, it can be used to estimate the outputs corresponding to the new inputs, for which the output values are not available. This is achieved in this project by the statistical method of Regression Analysis. 
Designing Regression computation model follows a number of systemic procedures. 
The general structure of developing a model in this project is divided into five basic steps:  (1) collecting data, (2) preprocessing data, (3) building the network, (4) training and (5) test performance of model. The basic flow in designing Regression model is given in Figure 1.  
Figure 2-1 Basic flow for designing Regression model.  
Data Collection: For the meteorological data source, weather data, is collected from (Meteorological Station) Department of Hydrology and Meteorology, Ministry of Energy, Water Resources and Irrigation. Department of Hydrology and Meteorology collect and disseminate hydrological and meteorological information for water resources, agriculture, energy, and other development activities. There are fifty-one Hydrology Station and two hundred eighty-two Meteorology Station all around Nepal through which data are collected.  
Data Collection
Preprocessing Data
Building network
Training network
Testing network
5  
For this project, the meteorological data of Kathmandu Airport, Nepal with global coordinates of longitude 85.3240° E, latitude 27.7172° N and elevation of 1337 meter, period from 2008 to 2017 has been chosen to build a regression model. 
The chosen weather data is divided into two groups, the training group, corresponding to 80% of the data, and the test group corresponding to 20% of data. Weather forecasts today depend on collecting and analyzing data and measurements from around the world. The data set contains five attributes. They are 
1) Maximum Temperature 2) Minimum Temperature 3) Mean Temperature 4) Precipitation 5) Relative Humidity  
Table 2.2: Data sheet of meteorological parameters of 1st week of January 
Date 
Mean temperature 
Max temperature 
Min temperature Precipitation  Relative Humidity 1/1/2008 10.8 20.3 1.3 0 100 1/2/2008 12.25 22.2 2.3 0 100 1/3/2008 12 22 2 0 100 1/4/2008 10.95 20.9 1 0 96.8 1/5/2008 11 22.2 -0.2 0 95.2 1/6/2008 9.3 17.6 1 0 93.5 1/7/2008 12.6 22.7 2.5 0 97.1  
     
Figure 2-2Time series plot of maximum temperature from year 2008-2009 
6   
Figure 2-3Time series plot of mean temperature from year 2007-2008  
Figure 2-4Time series plot of minimum  temperature from year 2007-2008   
Figure 2-5Time series plot of meteorological parameters 
7  
Preprocessing Data Building an effective regression model requires careful consideration of the network architecture as well as the input data format. So, the input data should not contain any missing field. The missing data field was solved by replacing the missing value by average of neighboring value. 
Theory 
Regression Analysis  Prediction essentially involves determining the functional relationships between the inputoutput variables of a system. Once such relationship is established from the available inputoutput data, it can be used to estimate the outputs corresponding to the new inputs for which output values are not available. The traditional method of empirical modeling is to assume some form of a fitting function with unknown parameters and subsequently employ the regression or optimization techniques for their estimation. The regression analysis is a branch of statistical theory that is widely used in almost all the scientific disciplines. [5] So the statistical tool with the help of which we are in a position to estimate (or predict) from known values of another variable is called regression. With the help of regression analysis, we are in a position to find out the average probable change in one variable given a certain amount of change in another. 
For the purpose of the regression analysis, we have used the term PREDICTOR (X) to refer to an antecedent parameter used in obtaining a forecast and the term PREDICTAND (Y) to refer to a weather element that is to be predicted. Here so many parameters (X1, X2,) are used. It always necessary to define the predict and in precise terms, the definition may have to be changed as study progress. If the values of the predictors at and prior to a given time are known, the problem is to derive some relationship (may it be an equation or diagram) which will provide a statement concerning the prediction at some future time. Ideally, this statement is, qualified as a final forecast. In many cases it will be found that it will serve only as approximation.  
The generalized formula for a Linear Regression model is:  
ŷ = β0 + β1 * x1 + β2 * x2 + ... + β(p-n) x(p-n) + Ε  
where: 
• ŷ is the predicted outcome variable (dependent variable) • xj are the predictor variables (independent variables) for j = 1,2, p-1 parameters 
8  
• β0 is the intercept or the value of ŷ when each xj equals zero • βj is the change in ŷ based on a one unit change in one of the corresponding xj • Ε is a random error term associated with the difference between the predicted ŷi value and the actual yi value 
Definition of the Predictand  Predictand is that which is to be predicted. In this project Mean Temperature is the predictand. It is essential that the predictand be defined in precise terms.  
Selection of predictor  This is most difficult part of the work and the result depends upon the choice. Although not essential, it is highly desirable that the predictor be chosen so as to represent variables, which have known physical or dynamical relationship to the predictand. In this project there are four parameters which are predictor, maximum temperature, minimum temperature, relative humidity, precipitation. 
 Selection and processing data  A key assumption required by the linear regression technique is that you have a linear relationship between the dependent variable and each independent variable. One way to assess the linearity between our independent variable, which for now will be the mean temperature, and the other independent variables is to calculate the Pearson correlation coefficient. 
The Pearson correlation coefficient (r) is a measurement of the amount of linear correlation between equal length arrays which outputs a value ranging -1 to 1. Correlation values ranging from 0 to 1 represent increasingly strong positive correlation. If two data series are positively correlated when values in one data series increase simultaneously with the values in the other series and, as they both go up in increasingly equal magnitude the Pearson correlation value will approach 1. 
Correlation values from 0 to -1 are said to be inversely, or negatively, correlated in that when the values of one series increase the corresponding values in the opposite series decrease but, as changes in magnitude between the series become equal (with opposite direction) the correlation value will approach -1. Pearson correlation values that closely straddle either side of zero are suggestive to have a weak linear relationship, becoming weaker as the value approaches zero. 
In this project, generally accepted set of classifications for the strengths of correlation is used and which is given below:  
9  
Table 2-1 Interpretion  of correlation value 
Correlation Value Interpretation 
0.8 - 1.0 Very Strong 
0.6 - 0.8 Strong 
0.4 - 0.6 Moderate 
0.2 - 0.4 Weak 
0.0 - 0.2 Very Weak  
In selecting features to include in this linear regression model, to reduce error on the side of being slightly less permissive in including variables with moderate or lower correlation coefficients.  So, removing the features that have correlation values less than the absolute value of 0.6 is done and the features that is included in the linear regression model now have the correlated value large than 0.6. 
Data should be selected for a period the length of which will depend upon the frequency of occurrence, the variability of attendant conditions and the number of predictors used. To obtain a relation between the predictors (X1, X2, X3) and the predictand (Y), the data may be combined by the standard statistical techniques into regression equations of the type.  
Y = a X1+ b X2 + c X3 + d X4 + ....... where a, b, c, d are some constants; X1, X2, X3  are predictor parameters. 
Occurrence and Non-Occurrence  Following the general procedure described above, one chooses two predictors (X1 and X2) who appear to have the closest bearing upon the problem. On occasion, and a pilot study may be needed to identify the most suitable predictors.  
For this scatter diagram of the different meteorological parameters is plotted for visual assist to verify that the meteorological parameters follow the pattern. Below scatter graph prove that there is in fact a linear relationship.   
10   
Fig: Scatter plot (Mean Temperature Vs Maximum Temperature)  
Fig: Scatter plot (Mean Temperature Vs Minimum Temperature) 
  
Fig: Scatter plot (Mean Temperature Vs Precipitation)   
11  
Fig: Scatter plot (Mean Temperature Vs Relative Humidity) 
Above fig show the scatter plot graph of Mean Temperature in y-axis and Maximum Temperature in x- axis and Mean Temperature in y-axis and Minimum Temperature in x-axis. 
From the scatter plots above it is recognizable that all the remaining predictor (maximum temperature and minimum temperature) variables show a good linear relationship with the response variable (Mean temperature). Additionally, above fig shows the relationships which is uniformly randomly distributed. A uniform random distribution of spread along the points is another important assumption of Linear Regression. Above fig shows that there is no linear relationship of predictor (relative humidity and precipitation) with response variable (Mean temperature). Due to this reason both predictors are not included as the predictor to build the model. Also, relative humidity and precipitation both shows the low correlated value. 
 Experiment The data file contains the meteorological parameters of Kathmandu Airport from year 2008 to year 2017. So, in this phase, regression analysis is done on the master data file. A predictand (Y) depends upon many predictors. To predict the mean temperature, mean temperature depends upon predictors or weather parameters like: 
• Rainfall • Humidity • Precipitation • Pressure • Wind speed and other 
In this project only 4 predictors are taken after the selection and processing of data into consideration. So we can write  
Predictand (Y) = some factor * predictor (X1) + some factor * predictor (X2) + some factor * predictor (X3) + .......  
Otherwise, it can be expressed as linear equation  
Y = a X1+ b X2 + c X3 + d X4 + ....... 
We have the values of the predictand Y1, Y2, Y3, Y4, for 2008 to 2017. They are the experimental values. So, by the above operation, the initial values of the coefficients are obtained. Using the above coefficients, the theoretical values of the predictand for the 
12  
corresponding years are obtained and we have the true values or experimental values of the predictand for corresponding year. It is found that the two are not matching. Our aim to adjust the coefficients in such a manner that it would fit for the data of all the years that means it holds true for the data for the coming year which will be calculated as  
Predictand (TEMP) = a*X, + b*X 2 + c*X 3 +d*X 4 . 
In many datasets there can be interactions that occur between variables that can lead to false interpretations of these simple hypothesis tests. To test for the effects of interactions on the significance of any one variable in a linear regression model a technique known as step-wise regression is often applied. Using step-wise regression in this project we add or remove variables from the model and assess the statistical significance of each variable on the resultant model. 
Backward elimination technique is used by including all the predictor and then removing the predictor that has no effect on the outcome variable's  
Backward Elimination works as follows: 
 Algorithm for Regression Analysis 1. Select a significance level Α for which you test your hypothesis against to determine if a variable should stay in the model  2. Fit the model with all predictor variables  3. Evaluate the p-values of the βj coefficients and for the one with the greatest p-value, if p-value > Α progress to step 4, if not you have your final model  4. Remove the predictor identified in step 3  5. Fit the model again but, this time without the removed variable and cycle back to step 3  Tools Required  All the programming is done in Python Programming Language by using the following tools and libraries.   
• Jupyter Notebook  • Numpy  • Pandas  • Mathplotlib • Statsmodels • sklearn 
13  
3. Result Analysis 
 Evaluation Metrics:  
The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For regression algorithms, three evaluation metrics are commonly used: 
The performance of all of the models was tested through the most common statistical indicators given below: 
Mean Absolute Error (MAE) The MAE is an average of the absolute errors which is the absolute difference between observed and estimated value. It gives an idea of how wrong the predictions were. It is calculated as:  
Mean Squared Error (MSE) : MSE is a weighted average of the squares of the difference between observed and estimated value of parameter.:  
Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:   
R square: R2 value is used to evaluate the whole simulation ability of our model, the R2 indicates how well our model can explain the raw data set. The R2 value of a model is: 
   
The Datasets is split in to training set and testing set.  80% of the data is used to train the model and remaining 20% is used to test the model. Table 4-1, shows the comparison between the actual mean temperature and predicted mean temperature of the five days with the error percentage. The average error percentage of the above test is around 4.46%, the negative sign indicates that the error is decreased overall.    
14  
Table 3-1Comparision of Actual Value vs Predicted Value 
Year Actual Value Predicted Value Error 8/27/2010 24.25 24.52633622 -1.13953 10/31/2009 19.4 19.31487861 0.43877 3/6/2013 19.5 19.76592271 -1.36371 5/14/2012 24.35 23.01798787 5.470276 4/24/2010 24.15 24.81364936 -2.74803  
Figure 3-1 show the time-series plot of the actual value of mean temperature and predicted value by the regression model. It can be deduced from the above figure 4-1, that the difference between blue lines (actual value) & orange lines (results by regression model) are very minimal, which signifies the precision and effectiveness of the regression model.   
Figure 3-1Comparision of Actual value (blue), predicted value by model (orange) 
Table 3-2Results of Evaluation metrics 
Performance Measure Data Result 
R2 0.955 
MSE 1.141  
RMSE 1.068 
MAE 0.797  
To predict mean temperature using regression model, about 80% of the data to train model and 20% to test the model As for the prediction of mean temperature, R2=0.955, i.e. the regression model explained about 95.5% of total variations, suggests that the regression model has high forecast accuracy and cannot predict only 4.5% of mean temperature variables. This indicates 
15  
the good performance of the model in predicting the mean temperature. Also, the mean square error and root mean square error is 1.141 and 1.068 respectively.        
  
16  
4.  CONCLUSIONS  
In this project, Linear Regression model was employed to forecast mean temperature for Kathmandu Airport. The model which is used is an efficient and accurate weather prediction and forecasting model using linear regression. All these concepts are a part of machine learning. The normal equation is a very efficient weather prediction model and using the mean temperature, minimum temperature, maximum temperature, humidity and precipitation, it can be used to make reliable weather predictions. This model also facilitates decision making in day to day life. It can yield better results when applied to cleaner and larger datasets. Preprocessing of the datasets can be effective in the prediction as unprocessed data can also affect the efficiency of the model. 
