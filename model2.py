import numpy
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

# model
# we are given a user and a business
# consider all businesses already rated by user
# consider all users who have also rated all these businesses
# select all users that have rated the given business
# look at how many businesses they have in common with the given user
# assign a weight depending on how close of a match their ratings are, between these users and the given user
# users that have a lot in common will be given higher weights
# calculate a weighted average of their ratings for the given business

# unique users: 41182
# unique businesses: 11874

train_reviews_file_name = 'train_reviews.csv'
validate_queries_file_name = 'validate_queries.csv'
test_queries_file_name = 'test_queries.csv'
max_index = 1000

class User():
    def __init__(self, user_id):
        self.user_id = user_id

class Business():
    def __init__(self, business_id):
        self.business_id = business_id
        self.visitors = []

def get_users():
    with open('users.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        users = []
        for row in csv_reader:
            users.append(User(row[20]))
    return users

def get_businesses():
    with open('business.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        businesses = []
        for row in csv_reader:
            businesses.append(Business(row[41]))
    return businesses

def train_reviews():
    with open(train_reviews_file_name, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        index = 0
        for row in csv_reader:
            if index < max_index:
                index += 1
                business_id = row[0]
                stars = row[5]
                user_id = row[8]
                for business in businesses:
                    if business.business_id == business_id:
                        business.visitors.append([user_id, stars])
                        break

def validate_queries():
    with open(validate_queries_file_name, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        square_error = 0
        prediction_count = 0
        for row in csv_reader:
            target_visitors = []
            stars_total = 0
            user_id = row[1]
            business_id = row[2]
            stars = float(row[3])
            for business in businesses:
                if business.business_id == business_id:
                    for visitor in business.visitors:
                        target_visitors.append(visitor)
                        stars_total += float(visitor[1])
                    break
            if target_visitors:
                prediction_count += 1
                # print(target_visitors, stars_total / len(target_visitors), round(stars_total / len(target_visitors),1), stars)
                square_error += numpy.square(round(stars_total / len(target_visitors),1) - stars)
        return square_error / prediction_count

def test_queries():
    with open(test_queries_file_name, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        index = 0
        for row in csv_reader:
            target_visitors = []
            stars_total = 0
            user_id = row[0]
            business_id = row[1]
            for business in businesses:
                if business.business_id == business_id:
                    for visitor in business.visitors:
                        target_visitors.append(visitor)
                        stars_total += float(visitor[1])
                    break
            # if target_visitors:
                # print(target_visitors, stars_total / len(target_visitors))

print('retrieving users')
users = get_users()
print('retrieving businesses')
businesses = get_businesses()
print('training')
train_reviews()

# for business in businesses:
#     if business.visitors:
#         print(business.visitors)

print('validating')
mse = validate_queries()
print('mse: {}'.format(mse))
print('testing')
test_queries()