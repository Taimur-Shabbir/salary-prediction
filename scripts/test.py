# write script that trains model on entire training set, saves model to disk,
# and scores the "test" dataset

# load test feature data

test_data = pd.read_csv("data/test_features.csv")

# perform same feature engineering on test features as on train features

# encode ordinal variables
test_data['Degree Category'] = test_data['degree']  # degree variable

test_data = test_data.replace({'Degree Category':
                               {'NONE': 0, 'HIGH_SCHOOL': 1,
                                'BACHELORS': 2, 'MASTERS': 3,
                                'DOCTORAL': 4}}, inplace=True)

test_data['Job Type Category'] = test_data['jobType']  # job type variable

test_data = test_data.replace({'Job Type Category':
                               {'JANITOR': 0, 'JUNIOR': 1,
                                'SENIOR': 2, 'MANAGER': 3,
                                'VICE_PRESIDENT': 4, 'CTO': 5,
                                'CFO': 6, 'CEO': 7}}, inplace=True)

# encode nominal variables
major_dummy_data = pd.get_dummies(test_data['major'])  # major variable

for column in major_dummy_data.columns:
    major_dummy_data.rename(
                           columns = {column: 'major_' + str(column)}, inplace=True)

test_data = pd.concat([test_data, major_dummy_data], axis=1)



industry_dummy_data = pd.get_dummies(test_data['industry'])  # industry variable

for column in industry_dummy_data.columns:
    industry_dummy_data.rename(
                              columns = {column: 'industry_' + str(column)}, inplace=True)

test_data = pd.concat([test_data, industry_dummy_data], axis=1)


# drop unnecessary features on test set

X_test = test_data.drop(['jobId', 'companyId',
                         'jobType', 'degree', 'major',
                         'industry'], axis=1)


# write function to train model on train data then make predictions on test data

def train_test_model(model, X_train, y_train, X_test, y_test):


    X_train = X_train.drop(['salary', 'jobId', 'companyId',
                            'jobType', 'degree', 'major',
                            'industry'], axis=1)


    # initialise tuned model
    model = GradientBoostingRegressor(n_estimators=160, learning_rate=0.1,
                                      max_depth=4, max_features=10,
                                      min_samples_split=1000)

    # fit model on X_train and y_train
    model.fit(X_train, y_train)

    # predict y_predicted using trained model
    y_predicted = model.predict(X_test)

    # test model and print mse
    mse = metrics.mean_squared_error(y_test, y_predicted)

    print('The MSE score on the test set is {}'.format(mse))
