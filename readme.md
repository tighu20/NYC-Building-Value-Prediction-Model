Based on the data available at https://opendata.cityofnewyork.us/ I have trained a supervised Machine Learning model to predict future 
future prices of NYC building. I have used XGBoost, CatBoost & Light GBM and compared thier results to select the best model suited for this task and performed hyper parameter optimzation using HyperOPT. In addition to this I used SHAP values to draw infrence from these models and conclude which values most impact a market value of a building in New York.

In this project my aim was to predict the price of buildings in New York City. The data is provided by the Department of Finance and is publicly available the NYC Open Data website (https://data.cityofnewyork.us/) . The data contains 9 million entries and 39 features containing data dated from 2010 and was last updated in June 2019. The aim was to predict the total value of the
building based on features such as location (latitude, longitude and borough), tax class, no of stories in the building, year etc. Due to memory issues later discussed in this paper the data was reduced to 6 million entries only using the data dated 2014 and later as the training set. The predictions were performed on the 2019 data. 

## Data Engineering

For this project I did extensive data engineering to make the data consumable by the model whereas on other scenarios it was to reduce the dimensionality of our data sets so that it can be managed by our current hardware capabilities.Explained below are how applied Data Engineering to the various columns:
- Year: Used regular expressions to parse year from original column and used it split data between training and testing (train:2014-2018, test:2019 )
- Easement Code: One hot encoding this column would have resulted in around 20 new columns considerably increasing the size data frame occupied in memory so I bundled the categories which were similar to each other.
- Exemption Code 1 and 2: These two columns had lots of categorical values (upwards of 1000) and therefore one hot encoding these columns was not practical even more so considering the number of rows in the data set (approx 6 million). Majority of categories in these columns had very little number of rows for each of them(1-10) so we segregated the columns having majority of rows (80-90 percent) which were around 40 or 50. This allowed us to capture enough variance and keep the size of data frame in check.
- I one hot encoded the categorical columns where the above situations were not applicable. Some numerical columns(Building front and depth) had zero values like instead of null values so I fixed that.
- I dropped columns which I believed would have no impact on deciding the value of a building(Lot number,Owner,Community Board,Council District,BIN,NTA)
- Apply all the above steps to the test data frame as well and removing the columns which are not common to both.

## Techniques Used

The main models wI used in our project are:
- XGBoost
- Light GBM
- CAT Boost


Some of the reasons for using these models over other trivial or base models like regression,random forest were:Our data had lots of missing values and I would have to manually impute those values using some imputation techniques like KNN,mean imputation etc. The above mentioned models can automatically handle such missing data and their in-built imputation techniques have proved out to be better than manual imputation. The above mentioned boosting algorithms have built-in L1 and L2 regularisation parameters which prevents the model from over fitting.Since these are advances boosting algorithms, they can utilize the power of parallel processing and that is why it is much faster than normal GBM.This was very useful for us as our data was too big and I had to use multiple cores to execute our
model.I can also run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run.A normal GBM would stop splitting a node when it encounters a negative loss in the split. Thus it is more of a greedy algorithm. XGBoost on the other hand make splits up to the max depth specified and then start pruning the tree backwards and remove splits beyond which there is no positive gain.

## Issues Faced
Majority of the issues faced while doing this project mainly because of the sheer size of dataset having the price data of NYC buildings from the past 10 years(10 million rows).I firstly subset data from original source to almost 6 million rows so I could work on this dataframe using the jupyter notebook as the complete dataset was often leading to the kernel going into a hung state and consecutively crashing due to memory failure.Even with the smaller version I had to use Windows advanced configuration options to use part of SSD(Solid State Disk) HDD(Hard Disk) as physical memory along with multiple uses of garbage collector function. Likewise many of our data engineering efforts bore out of the need for dimension reduction in our data frame.The number of columns in the dataframe after the necessary data engineering to make it consumable my gradient boosted trees models was 111. A sizeable number of columns even though it was sparse matrix considering the many categorical columns combined with almost 6 million rows made for a slow modeling process. For example an average hyperopt cross validation run for XGB took upwards of 30 hours (relatively small parameter sample space) and with all threads of processor occupied by the kernel it would render the system unusable for anything other than that (Full CPU usage). All these things made for an extensive hyper parameter tuning period. LightGBM had a much faster runtime comparatively showacasing a chink in the workings of XGB when it comes to handling high dimensions as it scan all rows to estimate information gain where ever splits occur (Week6). Light GBM saves on memory by converting continuous values into bundles greatly reducing its runtime.It also improved from the performance of XGB both in terms of reduction of RMSE and increase in R2 score. Catboost was behind the other two GBT in terms of performance metrics.

## Results
With XGB I got a test R2 score of 0.94 and an RMSE of 3.5 million.With Catboost I got an RMSE score of 3.1 million. Light GBM performed best with an RMSE of 2.7 million and test R2 score of 0.97.The high value of RMSE may suggest a model performing way underpar but this is a really good score considering values of some of the buildings in New York bourgh go as high as tens of billions.

## Conclusions
This model was successful is predicting building values in New York City using its other features, and can be used to predict values of the future beyond 2019, if given the necessary information. The LightGBM model gives the best result out of the three gradient boosting methods used in this project. The most important features according to my model are :
- AVTOT : Transitional Total Land
- TAXCLASS 1: The Tax class under which the building comes
- LATITUDE
- LONGITUDE
- POSTCODE
- AVLAND2: Transitional Total Land Value

That being said my model does have scope for improvement. Due to memory size limitations I had to group certain
categories together to reduce the number of features for one-hot encoding. Due to memory issues again the data size
had to be reduced to 6 million entries instead of 10 million. Larger compute resources can be used to tackle this data
mining problem in a better way. NYC Open Data also has lots of other data sets available, I can look to integrate this data with other meaningful datasets available and get significant insights.

## References
- XGBoost Documentation. https://xgboost.readthedocs.io/en/latest/index.html.
- LightGBM Documentation. https://lightgbm.readthedocs.io/en/latest/.
- Cat Boost Documentaion. https://catboost.ai/.
- Open NYC Data - Dataset source. https://data.cityofnewyork.us/City-Government/Property-Valuation-and-
Assessment-Data/yjxr-fw8i.
- NYC Building Data Documentaion. https://www1.nyc.gov/assets/finance/jump/hlpbldgcode.html.
