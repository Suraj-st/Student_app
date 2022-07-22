import pandas as pd
import numpy as np
from sklearn import preprocessing

student = pd.read_csv(r'dataset.csv', sep=";")

drop_cols = ['Application mode','Course','Previous qualification','Nacionality',"Mother's qualification",
             "Father's qualification", "Mother's occupation","Father's occupation"]
student.drop(drop_cols, axis = 1, inplace = True)

student['Curricular enrolled_u per sem'] = (student['Curricular units 1st sem (enrolled)'] + student['Curricular units 2nd sem (enrolled)'])/2
student['Curricular evaluation_u per sem'] = (student['Curricular units 1st sem (evaluations)'] + student['Curricular units 2nd sem (evaluations)'])/2
student['Curricular approved_u per sem'] = (student['Curricular units 1st sem (approved)'] + student['Curricular units 2nd sem (approved)'])/2
student['Curricular grade_u per sem'] = (student['Curricular units 1st sem (grade)'] + student['Curricular units 2nd sem (grade)'])/2
student['Curricular w/o_evaluation_u per sem'] = (student['Curricular units 1st sem (without evaluations)'] + student['Curricular units 2nd sem (without evaluations)'])/2

drop_colms = ['Curricular units 1st sem (credited)','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)',
             'Curricular units 1st sem (approved)',"Curricular units 1st sem (grade)","Curricular units 1st sem (without evaluations)",
             'Curricular units 2nd sem (credited)','Curricular units 2nd sem (enrolled)','Curricular units 2nd sem (evaluations)',
             'Curricular units 2nd sem (approved)','Curricular units 2nd sem (grade)','Curricular units 2nd sem (without evaluations)'
            ]
student.drop(drop_colms, axis = 1, inplace = True)

target = 'Target'
encode = ['Marital status', 'Daytime/evening attendance', 'Displaced', 'Educational special needs',
          'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']

for col in encode:
    dummy = pd.get_dummies(student[col], prefix=col)
    student = pd.concat([student, dummy], axis=1)
    del student[col]

drop_dcols = ['Marital status_6','Daytime/evening attendance_0','Displaced_0','Educational special needs_0','Debtor_0',
              'Tuition fees up to date_0','Gender_0','Scholarship holder_0','International_0']
student.drop(drop_dcols, axis = 1, inplace = True)

target_mapper = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}


def target_encode(val):
    return target_mapper[val]


student['Target'] = student['Target'].apply(target_encode)

columns = ['Application order','Age at enrollment','Unemployment rate','Inflation rate','GDP','Curricular enrolled_u per sem',
           'Curricular evaluation_u per sem','Curricular approved_u per sem','Curricular grade_u per sem',
           'Curricular w/o_evaluation_u per sem','Marital status_1','Marital status_2','Marital status_3','Marital status_4',
           'Marital status_5','Daytime/evening attendance_1','Displaced_1','Educational special needs_1','Debtor_1',
           'Tuition fees up to date_1','Gender_1','Scholarship holder_1','International_1']
X_data= student[columns]


# X_data = student.drop('Target', axis=1)
Y = student['Target']

min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))

# X = min_max_scaler.fit_transform(X_data)

# from xgboost import XGBClassifier
# params_xgb = {'n_estimators':50, 'max_depth':5}
# model_xgb = XGBClassifier(**params_xgb)
# model_xgb.fit(X,Y)

from sklearn.ensemble import RandomForestClassifier
params_rf = {'bootstrap': True,
             'max_depth': 32,
             'max_features': 9,
             'min_samples_leaf': 6,
             'min_samples_split': 14,
             'n_estimators': 128}
model_rf = RandomForestClassifier(**params_rf)
model_rf.fit(X_data, Y)


import pickle
pickle.dump(model_rf, open('student_rf.pkl', 'wb'))












