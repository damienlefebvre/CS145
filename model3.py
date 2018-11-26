import numpy
import pandas as pd
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

import sys
import csv

maxInt = sys.maxsize
decrement = True

while decrement:
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

# retrieve the current date
current_date = datetime.datetime.now()

def get_users():
    with open('users.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        users = {}
        count = 0
        for row in csv_reader:
            user_id = row[20]
            if not user_id in users:
                users[user_id] = count
                count += 1
    return users

def get_businesses():
    with open('business.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        businesses = {}
        count = 0
        for row in csv_reader:
            business_id = row[41]
            if not business_id in businesses:
                businesses[business_id] = count
                count += 1
    return businesses

# create maps
print('retrieve users')
users = get_users()
print('retrieve businesses')
businesses = get_businesses()

# open train_reviews
train_reviews_df = pd.read_csv("train_reviews_1000.csv", header=0)

# delete columns: review_id, text
del train_reviews_df['review_id'], train_reviews_df['text']

# convert from string to datetime
train_reviews_df.date = pd.to_datetime(train_reviews_df.date)

# convert from date to amount of time before current_date
train_reviews_df.date = current_date - train_reviews_df.date

# convert from amount of time to number of days
train_reviews_df.date = train_reviews_df.date.dt.days

# convert user_id and business_id using maps
print('convert ids')
for index, row in train_reviews_df.iterrows():
    train_reviews_df.loc[index, 'user_id'] = users[row['user_id']]
    train_reviews_df.loc[index, 'business_id'] = businesses[row['business_id']]

# delete columns: cool, date, funny, useful
del train_reviews_df['cool'], train_reviews_df['date'], train_reviews_df['funny'], train_reviews_df['useful']

# split train_reviews_df into features and training
x_train = train_reviews_df.drop(['stars'], axis=1)
y_train = train_reviews_df.stars

# reorder columns
reorder_col = ['user_id', 'business_id']
x_train = x_train[reorder_col]

# instantiate the model
linreg = LinearRegression()

# fit the model with data
print('fit linreg')
linreg.fit(x_train,y_train)

# # instantiate the model
# logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial')

# # fit the model with data
# print('fit logreg')
# logreg.fit(x_train,y_train)

# # predict on the training set
# print('predict logreg')
# y_pred_logreg = logreg.predict(x_train)

# open validate_queries
validate_queries_df = pd.read_csv("validate_queries_1000.csv", header=0)

# delete columns: unnamed
del validate_queries_df['Unnamed: 0']

# convert user_id and business_id using maps
print('convert ids')
for index, row in validate_queries_df.iterrows():
    validate_queries_df.loc[index, 'user_id'] = users[row['user_id']]
    validate_queries_df.loc[index, 'business_id'] = businesses[row['business_id']]

# split validate_queries_df into features and training
x_test = validate_queries_df.drop(['stars'], axis=1)
y_test = validate_queries_df.stars

# predict on testing set
print('predict linreg on x_test')
y_pred_linreg2 = linreg.predict(x_test)

square_error = 0
for index, row in y_test.iteritems():
    square_error += numpy.square(row - y_pred_linreg2[index])

rmse = numpy.sqrt(square_error / len(y_pred_linreg2))
print('rmse', rmse)