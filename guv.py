# Gender recognition using Voice
from __future__ import division,print_function
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from pandas import read_csv

"""(meanfreq","sd","median","Q25","Q75",
"IQR","skew","kurt","sp.ent","sfm","mode",
"centroid","meanfun","minfun","maxfun","meandom",
"mindom","maxdom","dfrange","modindx","label")

(0.0597809849598081,0.0642412677031359,0.032026913372582,
0.0150714886459209,0.0901934398654331,0.0751219512195122,
12.8634618371626,274.402905502067,0.893369416700807,
0.491917766397811,0,0.0597809849598081,0.084279106440321,
0.0157016683022571,0.275862068965517,0.0078125,0.0078125,0.0078125,0,0,
"male")

last value is the output. [0:20) are inputs;
"""
#csv_file = 'voice.csv'


def extract_csv(csv_file='voice.csv'):
    csv_data = read_csv(csv_file)
    feature_names,arr = list(csv_data.keys()),csv_data.values

    d = {'male': 0, 'female': 1}

    [train, test] = train_test_split(arr, test_size=0.3)

    [train_data, train_target_names] = train[:, range(0, 20)], train[:, -1]
    [test_data, test_target_names] = test[:, range(0, 20)], test[:, -1]

    train_target = np.array(list(d[i] for i in train_target_names))
    test_target = np.array(list(d[i] for i in test_target_names))

    train_set_r = {'data': train_data, 'target': train_target,
                 'target_names': train_target_names, 'feature_names': feature_names}
    test_set_r = {'data': test_data, 'target': test_target,
                'target_names': test_target_names, 'feature_names': feature_names}
    return train_set_r, test_set_r


if __name__ == '__main__':
    (train_set, test_set) = extract_csv()
    #clf = svm.SVC(kernel='linear',C=30)
    clf = svm.SVC(gamma=0.40,C=90)
    clf.fit(train_set['data'], train_set['target'])
    pr = clf.predict(test_set['data'])
    acc_sc = accuracy_score(test_set['target'],pr)
    print("Test accuracy: ",acc_sc)

    pr = clf.predict(train_set['data'])
    print("Train Accuracy: ",accuracy_score(train_set['target'],pr))

