#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','long_term_incentive','deferred_income',
                             'deferral_payments','loan_advances','other','expenses','director_fees',
                             'total_payments','exercised_stock_options','restricted_stock',
                             'restricted_stock_deferred','total_stock_value','from_messages','to_messages',
                             'fraction_from_poi','fraction_to_poi','shared_receipt_with_poi']
                         

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
features = data_dict['METTS MARK'].keys()

# Creating a data frame 
import pandas as pd
data_dict_df = pd.DataFrame(data_dict)
data_dict_df =pd.DataFrame.transpose(data_dict_df)

# Viewing the data frame
print  data_dict_df.head()

# Rearranging columns for easy readability 
data_dict_df = data_dict_df[['poi','salary','bonus','long_term_incentive','deferred_income',
                             'deferral_payments','loan_advances','other','expenses','director_fees',
                             'total_payments','exercised_stock_options','restricted_stock',
                             'restricted_stock_deferred','total_stock_value','email_address','from_messages','to_messages',
                             'from_poi_to_this_person','from_this_person_to_poi','shared_receipt_with_poi']]

# Droppping email address column 
data_dict_df = data_dict_df.drop('email_address', 1)

#Total number of data points
print len(data_dict_df)

#Total number of POI's and non POI's 
print data_dict_df['poi'].value_counts()

# Changing data to suitable datatype
data_dict_df.bonus = data_dict_df.bonus.astype(float)
data_dict_df.deferral_payments = data_dict_df.deferral_payments.astype(float)
data_dict_df.deferred_income = data_dict_df.deferred_income.astype(float)
data_dict_df.director_fees = data_dict_df.director_fees.astype(float)
data_dict_df.exercised_stock_options = data_dict_df.exercised_stock_options.astype(float)
data_dict_df.expenses = data_dict_df.expenses.astype(float)
data_dict_df.from_messages = data_dict_df.from_messages.astype(float)
data_dict_df.from_poi_to_this_person = data_dict_df.from_poi_to_this_person.astype(float)
data_dict_df.from_this_person_to_poi = data_dict_df.from_this_person_to_poi.astype(float)
data_dict_df.loan_advances = data_dict_df.loan_advances.astype(float)
data_dict_df.long_term_incentive = data_dict_df.long_term_incentive.astype(float)
data_dict_df.other = data_dict_df.other.astype(float)
data_dict_df.poi = data_dict_df.poi.astype(int)
data_dict_df.restricted_stock = data_dict_df.restricted_stock.astype(float)
data_dict_df.restricted_stock_deferred = data_dict_df.restricted_stock_deferred.astype(float)
data_dict_df.salary = data_dict_df.salary.astype(float)
data_dict_df.shared_receipt_with_poi = data_dict_df.shared_receipt_with_poi.astype(float)
data_dict_df.to_messages = data_dict_df.to_messages.astype(float)
data_dict_df.total_payments = data_dict_df.total_payments.astype(float)
data_dict_df.total_stock_value = data_dict_df.total_stock_value.astype(float)


#Total number of nulls in each column
print data_dict_df.isnull().sum()


#Summary of the data
print data_dict_df.describe()

### Task 2: Outlier Removal

#Dropping some rows that are not relavent 
data_dict_df = data_dict_df.dropna(thresh=2)
data_dict_df = data_dict_df.drop(['TOTAL'])
data_dict_df = data_dict_df.drop(['THE TRAVEL AGENCY IN THE PARK'])

#Filling nans with zero's
data_dict_df.update(data_dict_df[['salary','bonus','long_term_incentive','deferred_income',
              'deferral_payments','loan_advances','other','expenses','director_fees',
              'total_payments','exercised_stock_options','restricted_stock','restricted_stock_deferred',
              'total_stock_value','from_messages','to_messages','from_poi_to_this_person',
              'from_this_person_to_poi','shared_receipt_with_poi']].fillna(0))

#checking for correctness in total_payments column
def total_payments_correctness(row):
    sum = row[1]+row[2]+row[3]+row[4]+row[5]+row[6]+row[7]+row[8]+row[9]
    if (sum != row[10]):
        return row
data_total_payments_correctness = data_dict_df.apply(total_payments_correctness, axis=1)
print data_total_payments_correctness        
        
# Dropping data points that are incorrect
data_dict_df = data_dict_df.drop(['BELFER ROBERT', 'BHATNAGAR SANJAY'])

# After wrangling total datapoints available  
print len(data_dict_df)
print data_dict_df['poi'].value_counts()
    

### Task 3: Create new feature(s)

def computeFraction( poi_messages, all_messages ):
    fraction = 0
    if poi_messages == "NaN" or all_messages == "NaN":
        fraction = 0
    else:
        fraction = (poi_messages)/(all_messages)
   
    return fraction
fraction_from_poi = []
fraction_to_poi = []
for index, row in data_dict_df.iterrows():
    
    from_poi_to_this_person = row["from_poi_to_this_person"]
    to_messages = row["to_messages"]
    compute_fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    fraction_from_poi.append(compute_fraction_from_poi)
    
    from_this_person_to_poi = row["from_this_person_to_poi"]
    from_messages = row["from_messages"]
    compute_fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    fraction_to_poi.append(compute_fraction_to_poi)

data_dict_df['fraction_from_poi'] = fraction_from_poi  
data_dict_df['fraction_to_poi'] = fraction_to_poi 
data_dict_df.update(data_dict_df[['fraction_from_poi','fraction_to_poi']].fillna(0))


### Store to my_dataset for easy export below.
data_dict_df1 =pd.DataFrame.transpose(data_dict_df)
my_dataset = data_dict_df1


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# splitting training and testing data
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    

# Training data with a classifier
import sklearn.pipeline
import sklearn.grid_search
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

select = SelectKBest()
clf = DecisionTreeClassifier()
#clf = RandomForestClassifier()
#clf = GaussianNB()
#clf = AdaBoostClassifier()

#('decision_tree', clf)
#('random_forest', clf)
#('gaussianNB', clf)
steps = [('feature_selection', select),
        ('decision_tree', clf)]

pipeline = sklearn.pipeline.Pipeline(steps)

#decision_tree__min_samples_split=[20,30,50],
#decision_tree__criterion=['entropy']

#random_forest__n_estimators=[40,50,60,70],
#random_forest__min_samples_split=[10,20,30]

parameters = dict(feature_selection__k=[13,15,19],
                 decision_tree__min_samples_split=[10,20,30,50],
                 decision_tree__criterion=['entropy'])

cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid=parameters)

cv.fit(features_train, labels_train)
labels_prediction = cv.predict(features_test)
report = sklearn.metrics.classification_report( labels_test, labels_prediction) 
print cv.best_params_                                              
print report                  
    
#Testing
test_classifier(cv, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
