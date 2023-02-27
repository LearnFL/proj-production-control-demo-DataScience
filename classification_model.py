import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import numpy as np

model = pd.read_excel(r"C:\Users\......\growth-model.xlsx")
data_eval = pd.read_excel(r"C:\Users\......\data.xlsx")
model_col_to_drop = ["COLUMNS_THAT_YOU_NEED_DROPPED"] 
model_col_drop_answers = "NAME_OF_THE_COLUMN_THAT_HAS_ANSWERS"
hot_encod_col = 'COLLUMN_THAT_NEEDS_HOT_ENCODED'
model_target = model_col_drop_answers

model = pd.DataFrame(model)
print(model)

enc = OneHotEncoder()

enc.fit(model)
encoder_df = pd.DataFrame(enc.fit_transform(model[[hot_encod_col]]).toarray())#.reshape(-1,1)
model = model.join(encoder_df)

model.drop(columns=model_col_to_drop, axis=1, inplace=True)
print('Model: ')
print(model)

model_sample =model.copy()
model_sample.drop(columns=[model_col_drop_answers], axis=1, inplace=True) 
print('Model sample: ')
print(model_sample)

X_train, X_test, y_train, y_test = train_test_split(model_sample, model[model_target], random_state=11)
print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

knn = KNeighborsClassifier()
gaus = GaussianNB()
svc = SVC()

data_eval = pd.DataFrame(data_eval)
print('Target: ')
print(data_eval)

enc.fit(data_eval)
encoder_df_target = pd.DataFrame(enc.fit_transform(data_eval[[hot_encod_col]]).toarray())#.reshape(-1,1)

data_target = data_eval.join(encoder_df_target)
print('Encoded target: ')
print(encoder_df_target)

data_target.drop(columns="stats", axis=1, inplace=True)
print('Target after drop')
print(data_target)

print('GaussianNB: ')
gaus.fit(X=X_train, y=y_train)
gaus_predicted = gaus.predict(X=data_target)
gaus_expected = y_test
gaus_expected = np.array(gaus_expected)# expected.reshape(-1,-1)

print(f"Predicted GaussianNB: {gaus_predicted}")
print(f"Expected GaussianNB: {gaus_expected}")
print(f"{gaus.score(X_test,y_test):.2%}")

print('KNN: ')
knn.fit(X=X_train, y=y_train)
knn_predicted = knn.predict(X=data_target)
knn_expected = y_test
knn_expected = np.array(knn_expected)# expected.reshape(-1,-1)

print(f"Predicted KNN: {knn_predicted}")
print(f"Expected KNN: {knn_expected}")
print(f"{knn.score(X_test,y_test):.2%}")

with open(r"C:\Users\...........Desktop\Classification.txt", 'w') as txt:
    txt.write("Model Training And Fitting \n")
    txt.write("GaussianNB is the best for Quality\n")
    txt.write("\n")

    txt.write(f"Predicted GaussianNB: \n {gaus_predicted} \n")
    txt.write("\n")
    txt.write(f"Score: {gaus.score(X_test,y_test):.2%} \n")
    txt.write("\n")
    
    txt.write(f"Predicted KNN: \n {knn_predicted} \n")
    txt.write("\n")
    txt.write(f"Score: {knn.score(X_test,y_test):.2%} \n")
