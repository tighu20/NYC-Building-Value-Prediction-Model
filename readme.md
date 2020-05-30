Based on the data available at https://opendata.cityofnewyork.us/ I have trained a supervised Machine Learning model to predict future 
future prices of NYC building. I have used XGBoost, CatBoost & Light GBM and compared thier results to select the best model suited for this task and performed hyper parameter optimzation using HyperOPT. In addition to this I used SHAP values to draw infrence from these models and conclude which values most impact a market value of a building in New York.

In this project my aim was to predict the price of buildings in New York City. The data is provided by the Department of Finance and is publicly available the NYC Open Data website (https://data.cityofnewyork.us/) . The data contains 9 million entries and 39 features containing data dated from 2010 and was last updated in June 2019. The aim was to predict the total value of the
building based on features such as location (latitude, longitude and borough), tax class, no of stories in the building, year etc. Due to memory issues later discussed in this paper the data was reduced to 6 million entries only using the data dated 2014 and later as the training set. The predictions were performed on the 2019 data. 

## Data Engineering

For this project I did extensive data engineering to make the data consumable by the model whereas on other scenarios it was to reduce the dimensionality of our data sets so that it can be managed by our current hardware capabilities.Explained below are how applied Data Engineering to the various columns:
-Year: Used regular expressions to parse year from original column and used it split data between training and testing (train:2014-2018, test:2019 )
-Easement Code: One hot encoding this column would have resulted in around 20 new columns considerably increasing the size data frame occupied in memory so I bundled the categories which were similar to each other.
-Exemption Code 1 and 2: These two columns had lots of categorical values (upwards of 1000) and therefore one hot encoding these columns was not practical even more so considering the number of rows in the data set (approx 6 million). Majority of categories in these columns had very little number of rows for each of them(1-10) so we segregated the columns having majority of rows (80-90 percent) which were around 40 or 50. This allowed us to capture enough variance and keep the size of data frame in check.
-I one hot encoded the categorical columns where the above situations were not applicable. Some numerical columns(Building front and depth) had zero values like instead of null values so I fixed that.
-I dropped columns which I believed would have no impact on deciding the value of a building(Lot number,Owner,Community Board,Council District,BIN,NTA)
-Apply all the above steps to the test data frame as well and removing the columns which are not common to both.
