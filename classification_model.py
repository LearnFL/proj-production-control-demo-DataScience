import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

# BLOCK BUILD DF

model = pd.read_excel(
    r"C:\Users\...\growth-model.xlsx")
data_eval = pd.read_excel(
    r"C:\Users\...\Z008-0137.xlsx")

# pd.set_option('display.max_columns', None)

model_col_to_drop = ["type", "run", "stats", "reject", "avg_ec", "std_ec", "mac_ec",
                     "q_max", "q_avg", "q_min", "q_bin", "avg_size", "g_rate", "yield", "quality"]
model_col_drop_answers = "g_rate_bin"
hot_encod_col = 'stats'
model_target = "g_rate_bin"


# BLOCK BUILD TRAIN MODEL

model = pd.DataFrame(model)
print(model)

enc = OneHotEncoder()

enc.fit(model)
encoder_df = pd.DataFrame(enc.fit_transform(
    model[[hot_encod_col]]).toarray())  # .reshape(-1,1)
model = model.join(encoder_df)

###########################################
model.columns = model.columns.astype(str)
###########################################

model.drop(columns=model_col_to_drop, axis=1, inplace=True)
print('Model: ')
print(model)

model_train = model.copy()
model_train.drop(columns=[model_col_drop_answers],
                 axis=1, inplace=True)  # 'q_bin', 'g_rate_bin'
print('Model sample: ')
print(model_train)

model_train_answers = model[model_target]


# BLOCK NORMOLIZERS
stdsc = StandardScaler()
mms = MinMaxScaler()
rbs = RobustScaler()
norm = Normalizer(norm='l2')


# BLOCK REDUCTION
pca = PCA(n_components=7)
lda = LDA(n_components=2)


# BLOCK TRAINING

X_train, X_test, y_train, y_test = train_test_split(
    model_train, model_train_answers, test_size=0.25, random_state=11)
print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# BLOCK DATA TO EVALUATE
data_eval = pd.DataFrame(data_eval)
print('Target: ')
print(data_eval)

enc.fit(data_eval)

encoder_df_target = pd.DataFrame(enc.fit_transform(
    data_eval[[hot_encod_col]]).toarray())  # .reshape(-1,1)

############################################################
data_target = data_eval.join(encoder_df_target)
############################################################

data_target.columns = data_target.columns.astype(str)
print('Encoded target: ')
print(encoder_df_target)

data_target.drop(columns="stats", axis=1, inplace=True)
print('Target after drop')
print(data_target)


# BLOCK CLASSIFIERS
knn = KNeighborsClassifier()
gaus = GaussianNB()
svc = SVC()
forest = RandomForestClassifier(
    criterion='gini', n_estimators=100, max_depth=5)
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(100, ), random_state=1)


# BLOCK FEATURE IMPORTANCE
ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X_train, y_train)
importance = np.abs(ridge.coef_)
feature_names = np.array(data_target.columns)
plt.bar(height=importance, x=feature_names)
plt.title("Feature importances via coefficients")
plt.show()

# BLOCK PIPE
pipe_forest = make_pipeline(rbs, forest)
pipe_knn = make_pipeline(rbs, lda, knn)
pipe_svc = make_pipeline(mms, stdsc, svc)
pipe_mlp = make_pipeline(mms, mlp)

pipe_forest.fit(X_train, y_train)
pipe_knn.fit(X_train, y_train)
pipe_svc.fit(X_train, y_train)
pipe_mlp.fit(X_train, y_train)

print(f"Forest pipe score: {pipe_forest.score(X_test, y_test)}")
print(f"KNN pipe score: {pipe_knn.score(X_test, y_test)}")
print(f"SVC pipe score: {pipe_svc.score(X_test, y_test)}")
print(
    f"MLP pipe score: {pipe_mlp.score(X_test, y_test)} \n")


# BLOCK PREDICTION
print('KNN: ')
pipe_knn_predicted = pipe_knn.predict(X=data_target)
knn_expected = y_test
knn_expected = np.array(knn_expected)  # expected.reshape(-1,-1)
print(f"Predicted KNN: {pipe_knn_predicted}")
print(f"KNN {pipe_knn.score(X_test,y_test):.2%}")


print('\nKNN MLP Neural network models (supervised): ')
pipe_mlp_predicted = pipe_mlp.predict(X=data_target)
pipe_mlp_expected = y_test
knn_expected = np.array(pipe_mlp_expected)  # expected.reshape(-1,-1)
print(f"Predicted MLP: {pipe_mlp_predicted}")
print(
    f"MLP {pipe_mlp.score(X_test,y_test):.2%}")


print('\nFOREST: ')
pipe_forest_predicted = pipe_forest.predict(X=data_target)
forest_expected = y_test
forest_expected = np.array(forest_expected)  # expected.reshape(-1,-1)
print(f"Predicted FOREST: {pipe_forest_predicted}")
print(f"FOREST {pipe_forest.score(X_test,y_test):.2%}")


print('\nGaussianNB: ')
gaus.fit(X=X_train, y=y_train)
gaus_predicted = gaus.predict(X=data_target)
gaus_expected = y_test
gaus_expected = np.array(gaus_expected)  # expected.reshape(-1,-1)
print(f"Predicted GaussianNB: {gaus_predicted}")
print(f"GAUS {gaus.score(X_test,y_test):.2%}")

print('\nSVC: ')
pipe_svc_predicted = pipe_svc.predict(X=data_target)
svc_expected = y_test
svc_expected = np.array(svc_expected)  # expected.reshape(-1,-1)
print(f"Predicted SVC: {pipe_svc_predicted}")
print(f"SVC {pipe_svc.score(X_test,y_test):.2%}")

# BLOCK AVERAGE SCORE
knn_mean = np.mean(pipe_knn_predicted)
forest_mean = np.mean(pipe_forest_predicted)
mlp_mean = np.mean(pipe_mlp_predicted)
average = (knn_mean + forest_mean + mlp_mean) / 3

print("\n")
print(f"KNN + FOREST + MLP AVG {average:.2f}")

with open(r"C:\Users\...\Classification.txt", 'w') as txt:
    for name, pred, score in zip(
            ['FOREST', 'KNN', 'MLP Neural network models', 'GaussianNB', 'SVC'],
            [pipe_forest_predicted, pipe_knn_predicted, pipe_mlp_predicted,
                gaus_predicted, pipe_svc_predicted],
            ['{0:.2%}'.format(pipe_forest.score(X_test, y_test)), '{0:.2%}'.format(pipe_knn.score(X_test, y_test)),
             '{0:.2%}'.format(pipe_mlp.score(X_test, y_test)),
             '{0:.2%}'.format(gaus.score(X_test, y_test)), '{0:.2%}'.format(pipe_svc.score(X_test, y_test))]):

        txt.write(f"Predicted {name}: \n {pred} \n")
        txt.write("\n")
        txt.write(f"Score: {score} \n")
        txt.write("\n")

    txt.write(f"KNN + FOREST + MLP AVG {average:.2f}")
