import csv
import random
import numpy

test_queries_file_name = 'test_queries.csv'
validate_queries_file_name = 'validate_queries.csv'
max_submission_file_writer_index = 50100

class Business():
    def __init__(self, business_id, review_count, stars):
        self.business_id = business_id
        self.review_count = review_count
        self.stars = stars

def get_business_list():
    with open('business.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        businesses = []
        for row in csv_reader:
            businesses.append(Business(row[41], int(row[58]), float(row[59])))
    return businesses

submission_file = open('submission.csv', 'w')
submission_file_writer = csv.writer(submission_file)
submission_file_writer.writerow(['index', 'stars'])

def test_queries():
    submission_file_writer_index = 0
    with open(test_queries_file_name, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            user_id = row[0]
            business_id = row[1]
            [review_count, stars] = infoBusiness(business_id)
            margin = 5 - stars
            if review_count:
                review_count -= 99
                margin = margin / review_count
            predict = round(random.gauss(stars, margin),1)
            submission_file_writer.writerow([submission_file_writer_index, predict])
            submission_file_writer_index += 1

def validate_queries(businesses):
    submission_file_writer_index = 0
    with open(validate_queries_file_name, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        square_error = 0
        max_submission_file_writer_index_over_10 = max_submission_file_writer_index / 10
        for row in csv_reader:
            if submission_file_writer_index > max_submission_file_writer_index:
                break
            user_id = row[1]
            business_id = row[2]
            target = float(row[3])
            for business in businesses:
                if business.business_id == business_id:
                    review_count = business.review_count
                    stars = business.stars
                    break
            #print(business_id, review_count, stars)
            margin = 5 - stars
            if review_count:
                review_count -= 99
                margin = margin / review_count
            predict = round(random.gauss(stars, margin),1)
            submission_file_writer.writerow([submission_file_writer_index, predict])
            submission_file_writer_index += 1
            #print(submission_file_writer_index, target, predict)
            if not submission_file_writer_index % max_submission_file_writer_index_over_10:
                print(submission_file_writer_index)
            square_error += numpy.square(target - predict)
        return square_error / (submission_file_writer_index - 1)

businesses = get_business_list()
mse = validate_queries(businesses)
print('max_submission_file_writer_index: {}'.format(max_submission_file_writer_index))
print('mse: {}'.format(mse))
print('rmse: {}'.format(numpy.sqrt(mse)))
