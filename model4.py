import numpy
import pandas as pd
import datetime
import time
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

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
        
train_data_size = 2000
validate_queries_size = 1000
test_queries_size = 60000

# retrieve the current date
current_date = datetime.datetime.now()

class User():
    def __init__(self, average_stars, review_count):
        self.average_stars = average_stars
        self.review_count = review_count

class Business():
    def __init__(self, review_count, stars):
        self.review_count = review_count
        self.stars = stars

class Model():
    def get_users(self):
        with open('users.csv', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            self.users = {}
            for row in csv_reader:
                average_stars = row[0]
                review_count = row[18]
                user_id = row[20]
                if not user_id in self.users:
                    self.users[user_id] = User(average_stars, review_count)
            print('collected users')

    def get_businesses(self):
        with open('business.csv', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            self.businesses = {}
            for row in csv_reader:
                business_id = row[41]
                review_count = row[58]
                stars = row[59]
                if not business_id in self.businesses:
                    self.businesses[business_id] = Business(stars, review_count)
            print('collected businesses')

    def get_train_data(self):
        # open train_reviews.csv
        train_reviews_df = pd.read_csv("train_reviews.csv", header=0)

        # delete columns: review_id, text, cool, date, funny, useful
        del train_reviews_df['review_id'], train_reviews_df['text'], train_reviews_df['cool'], train_reviews_df['date'], train_reviews_df['funny'], train_reviews_df['useful']

        # truncate rows
        train_reviews_df = train_reviews_df.truncate(after=train_data_size)

        # create empty x_train with named rows
        self.x_train = pd.DataFrame(columns=['user_average_stars', 'user_review_count', 'business_review_count', 'business_stars'])

        # convert user_id and business_id using maps
        train_index = 0
        start = time.time()
        for index, row in train_reviews_df.iterrows():
            if not train_index % 1000:
                print('total rows:', train_index)
                print('last iteration:', '{0:.2f}'.format(time.time() - start))
                start = time.time()

            self.x_train.loc[index, 'user_average_stars'] = self.users[row['user_id']].average_stars
            self.x_train.loc[index, 'user_review_count'] = self.users[row['user_id']].review_count
            self.x_train.loc[index, 'business_review_count'] = self.businesses[row['business_id']].review_count
            self.x_train.loc[index, 'business_stars'] = self.businesses[row['business_id']].stars
            train_index += 1

        # split train_reviews_df into features and training
        self.y_train = train_reviews_df.stars
        
        print('collected train data')

    def prep_linreg(self):
        # instantiate the model
        linreg = LinearRegression()

        # fit the model with data
        linreg.fit(self.x_train,self.y_train)

        print('prepped linreg')

        return linreg

    def run_validate_queries(self, linear_model):
        # open validate_queries
        validate_queries_df = pd.read_csv("validate_queries.csv", header=0)

        # truncate rows
        validate_queries_df = validate_queries_df.truncate(after=validate_queries_size)

        # delete columns: unnamed
        del validate_queries_df['Unnamed: 0']
        
        # create empty x_test with named rows
        self.x_test = pd.DataFrame(columns=['user_average_stars', 'user_review_count', 'business_review_count', 'business_stars'])

        train_index = 0
        start = time.time()
        for index, row in validate_queries_df.iterrows():
            if not train_index % 1000:
                print('total rows:', train_index)
                print('last iteration:', '{0:.2f}'.format(time.time() - start))
                start = time.time()

            self.x_test.loc[index, 'user_average_stars'] = self.users[row['user_id']].average_stars
            self.x_test.loc[index, 'user_review_count'] = self.users[row['user_id']].review_count
            self.x_test.loc[index, 'business_review_count'] = self.businesses[row['business_id']].review_count
            self.x_test.loc[index, 'business_stars'] = self.businesses[row['business_id']].stars
            train_index += 1

        # extract stars from validate_queries
        self.y_test = validate_queries_df.stars

        # predict on testing set
        y_pred_linreg = linear_model.predict(self.x_test)

        square_error = 0
        for index, row in self.y_test.iteritems():
            square_error += numpy.square(row - y_pred_linreg[index])

        rmse = numpy.sqrt(square_error / len(y_pred_linreg))
        print('ran on validate queries, rmse:', rmse)

    def run_test_queries(self, linear_model):
        submission_file = open('submission.csv', 'w')
        submission_file_writer = csv.writer(submission_file)
        submission_file_writer.writerow(['index', 'stars'])

        # open test_queries
        test_queries_df = pd.read_csv("test_queries.csv", header=0)
        
        # truncate rows
        test_queries_df = test_queries_df.truncate(after=test_queries_size)
        
        # create empty x_predict with named rows
        self.x_predict = pd.DataFrame(columns=['user_average_stars', 'user_review_count', 'business_review_count', 'business_stars'], dtype=float)

        train_index = 0
        start = time.time()
        for index, row in test_queries_df.iterrows():
            if not train_index % 1000:
                print('total rows:', train_index, 'last iteration:', '{0:.2f}'.format(time.time() - start))
                start = time.time()
            
            user = self.users[row['user_id']]
            business = self.businesses[row['business_id']]
            self.x_predict.loc[index] = [user.average_stars, user.review_count, business.review_count, business.stars]

#             self.x_predict.loc[index] = [self.users[row['user_id']].average_stars, self.users[row['user_id']].review_count, self.businesses[row['business_id']].review_count, self.businesses[row['business_id']].stars]
            
#             self.x_predict.loc[index, 'user_average_stars'] = self.users[row['user_id']].average_stars
#             self.x_predict.loc[index, 'user_review_count'] = self.users[row['user_id']].review_count
#             self.x_predict.loc[index, 'business_review_count'] = self.businesses[row['business_id']].review_count
#             self.x_predict.loc[index, 'business_stars'] = self.businesses[row['business_id']].stars
            train_index += 1

        # predict on testing set
        self.y_predict = numpy.round(linear_model.predict(self.x_predict),0)
        
        for index, row in enumerate(self.y_predict):
            submission_file_writer.writerow([index, row])

        print('ran on test queries')

print('imported model4')
model = Model()
model.get_users()
model.get_businesses()
model.get_train_data()
linreg = model.prep_linreg()
model.run_validate_queries(linreg)
model.run_test_queries(linreg)