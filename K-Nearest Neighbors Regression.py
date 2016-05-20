import pandas as pd
import random
from numpy.random import permutation
import math
import os
from sklearn.neighbors import KNeighborsRegressor
from math import *

def main(featureFile, outputfolder):
    with open(featureFile, 'r') as csvfile:
        my_data = pd.read_csv(csvfile, delimiter="\t", low_memory=False)

    random_indices = permutation(my_data.index)
    # how many time do we want the data in our test set?
    test_cutoff = math.floor(len(my_data)/3)
    test = my_data

    # Generate the training set with the rest of the data.
    train = my_data.loc[random_indices[test_cutoff:]]

    x_columns = ["Row"=="1", "Student ID"=="2", "Problem Hierarchy" == "3", "Problem Name"=="4", "Problem View" == "5", "Step Name" == "6",
            "KC(Default)"=="7", "Opportunity (Default)" == "8"]
    x_columns = [int(i) for i in x_columns]
    # y columns show the predicted feature, in this case, the correct first attempt
    y_column = ["Correct First Attempt"]

    # Look at the Ten closest neighbors, to offset potential noise in the data
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(train[x_columns], train[y_column])

    # Make point predictions on the test set using the fit model.
    predictions = knn.predict(test[x_columns])
    actual = test[y_column]
    result = test[['Anon Student Id','Correct First Attempt']]
    result.to_csv(outputfolder, sep='\t')

    # Compute the root mean squared error of our predictions.
    rmse = math.sqrt((((predictions - actual) ** 2).sum()) / len(predictions))
    print('RMSE=')
    print(rmse)

if __name__ == '__main__':
      # parser = argparse.ArgumentParser()
      # parser.add_argument("featureFile", help="filepath for the feature data ")
      # parser.add_argument("outputfolder", help="Output folder where Result will be outputted ")

    main("algebra_2005_2006_train.txt","Result.txt")

