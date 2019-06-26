import pandas as pd
from pandas import DataFrame
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("....Reading DataSet and Creating Pandas DataFrame....")
alphabet_data = pd.read_csv("/home/jasmeet/PycharmProjects/Character_Recognition/SVM_WithFoundDataset/A_Z Handwritten Data.csv")
print("...DataFrame Created...")


print("...Slicing and creating initial training and testing set...")
X_Train_A = alphabet_data.iloc[:13869, 1:]
Y_Train_A = alphabet_data.iloc[:13869, 0]
X_Train_C = alphabet_data.iloc[22537:45946, 1:]
Y_Train_C = alphabet_data.iloc[22537:45946, 0]
X_Train = pd.concat([X_Train_A, X_rain_C], ignore_index=True)
Y_Train = pd.concat([Y_Train_A, Y_Train_C], ignore_index=True)
print("...X_Train and Y_Train created...")



train_features, test_features, train_labels, test_labels = train_test_split(X_Train, Y_Train, test_size=0.25, random_state=0)

clf = SVC(kernel='linear')
print("")
print("...Training the Model...")
clf.fit(train_features, train_labels)
print("...Model Trained...")


labels_predicted = clf.predict(test_features)
print(test_labels)
print(labels_predicted)
accuracy = accuracy_score(test_labels, labels_predicted)

print("")
print("Accuracy of the model is:  ")
print(accuracy)

print("...Saving the trained model...")
joblib.dump(clf, "alphabet_classifier.pkl", compress=3)
print("...Model Saved...")