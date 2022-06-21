#/itmo_contest/preprocessing.py created by: Nikolay Pavlychev pavlychev.n.se@gmail.com
#-----------------------------------------------------------------------------------------------------------------------
print('Import libs...')
import os
import sys
import random
import time

import numpy as np
import scipy
import pandas as pd
import sklearn

print('Successfully!')
#-----------------------------------------------------------------------------------------------------------------------
print('Import data...')
ROOT_DIR = os.path.abspath(os.curdir)

cols_comp_students = ['ISU', 'KURS', 'DATE_START', 'DATE_END', 'PRIZNAK', 'MAIN_PLAN']
comp_students = pd.read_csv(ROOT_DIR + '/data/' + 'comp_students.csv', dtype=object,sep=',', header=1, names=cols_comp_students)

cols_comp_portrait = ['ISU', 'GENDER', 'CITIZENSHIP', 'EXAM_TYPE', 'EXAM_SUBJECT_1', 'EXAM_SUBJECT_2', 'EXAM_SUBJECT_3',
                      'ADMITTED_EXAM_1', 'ADMITTED_EXAM_2', 'ADMITTED_EXAM_3', 'ADMITTED_SUBJECT_PRIZE_LEVEL',
                      'REGION_ID']
comp_portrait = pd.read_csv(ROOT_DIR + '/data/' + 'comp_portrait.csv', dtype=object,sep=',', header=1, names=cols_comp_portrait)

cols_comp_marks = ['ISU', 'ST_YEAR', 'SEMESTER', 'TYPE_NAME', 'MARK', 'MAIN_PLAN', 'DISC_ID', 'PRED_ID']
comp_marks = pd.read_csv(ROOT_DIR + '/data/' + 'comp_marks.csv', dtype=object,sep=',', header=1, names=cols_comp_marks)

cols_comp_disc = ['PLAN_ID', 'DISC_ID', 'CHOICE', 'SEMESTER', 'DISC_NAME', 'DISC_DEP', 'KEYWORD_NAMES']
comp_disc = pd.read_csv(ROOT_DIR + '/data/' + 'comp_disc.csv', dtype=object,sep=',', header=1, names=cols_comp_disc)

cols_comp_teachers = ['ISU', 'GENDER', 'DATE_BIRTH', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'MAIN_PLAN', 'TYPE_NAME', 'MARK']
comp_teachers = pd.read_csv(ROOT_DIR + '/data/' + 'comp_teachers.csv', dtype=object,sep=',', header=1, names=cols_comp_teachers)

cols_train = ['ISU', 'ST_YEAR','SEMESTER', 'DISC_ID', 'TYPE_NAME', 'MARK', 'DEBT']
train = pd.read_csv(ROOT_DIR + '/samples/' + 'train.csv', dtype=object,sep=',', header=1, names=cols_train)

cols_test = ['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'TYPE_NAME']
test = pd.read_csv(ROOT_DIR + '/samples/' + 'test.csv', dtype=object,sep=',', header=1, names=cols_test)

print('Successfully!')
#-----------------------------------------------------------------------------------------------------------------------
print("Start preprocessing...")
print("Converting data type and print info...")
comp_students['DATE_START'] = pd.to_datetime(comp_students['DATE_START'],yearfirst=True).dt.normalize()
comp_students['DATE_END'] = pd.to_datetime(comp_students['DATE_END'],yearfirst=True).dt.normalize()

print(comp_students.info())

comp_portrait['ADMITTED_EXAM_1'] = comp_portrait['ADMITTED_EXAM_1'].astype(float)
comp_portrait['ADMITTED_EXAM_2'] = comp_portrait['ADMITTED_EXAM_2'].astype(float)
comp_portrait['ADMITTED_EXAM_3'] = comp_portrait['ADMITTED_EXAM_3'].astype(float)

print(comp_portrait.info())

print(comp_marks.info())

print(comp_disc.info())

comp_teachers['MARK']= comp_teachers['MARK'].astype(float)

print(comp_teachers.info())

train['DEBT'] = train['DEBT'].astype(int)

print(train.info())

print(test.info())

print('Successfully!')
#-----------------------------------------------------------------------------------------------------------------------
print('Merging data...')
print('comp_students table:')
print(' Full students ISU:',comp_students.shape[0],'\n',
      'Unique students ISU=',comp_students['ISU'].unique().shape[0])
print('Check duplicates:')
print(comp_students.shape[0]/comp_students.drop_duplicates().shape[0])

print('comp_portrait table:')
print(' Full students ISU:',comp_portrait.shape[0],'\n',
      'Unique students ISU=',comp_portrait['ISU'].unique().shape[0])
print('Check duplicates:')
print(comp_portrait.shape[0]/comp_portrait.drop_duplicates().shape[0])

comp_portrait_students = comp_portrait.merge(comp_students,on=['ISU'],how='outer')
print('comp_portrait_students shape = ',comp_portrait_students.shape)

print('comp_marks table:')
print(' Full students ISU:',comp_marks.shape[0],'\n',
      'Unique students ISU=',comp_marks['ISU'].unique().shape[0])
print('Check duplicates:')
print(comp_marks.shape[0]/comp_marks.drop_duplicates().shape[0])
comp_marks = comp_marks.drop_duplicates()
print('Check duplicates:')
print(comp_marks.shape[0]/comp_marks.drop_duplicates().shape[0])

print('comp_disc table:')
print('Check duplicates:')
print(comp_disc.shape[0]/comp_disc.drop_duplicates().shape[0])
comp_disc = comp_disc.drop_duplicates()
print('Check duplicates:')
print(comp_disc.shape[0]/comp_disc.drop_duplicates().shape[0])

print('comp_teachers table:')
print('Check duplicates:')
print(comp_teachers.shape[0]/comp_teachers.drop_duplicates().shape[0])
comp_teachers = comp_teachers.drop_duplicates()
print('Check duplicates:')
print(comp_teachers.shape[0]/comp_teachers.drop_duplicates().shape[0])

comp_disc = comp_disc.rename(columns={'PLAN_ID':'MAIN_PLAN'})
comp_disc_teachers = comp_disc.merge(comp_teachers,on=['MAIN_PLAN', 'DISC_ID'],how='outer')
print('comp_disc_teachers shape = ',comp_disc_teachers.shape)


#-----------------------------------------------------------------------------------------------------------------------
print('Preparing target...')

target = train[['ISU', 'DISC_ID', 'TYPE_NAME', 'DEBT']].drop_duplicates()
print(train.shape, target.shape)
isu_mark_history = train.groupby(by=['ISU', 'DISC_ID'])['DEBT'].agg({'median'})
students_mark_history_train = train[train['ST_YEAR'].isin(['2018','2019'])].groupby(by=['DISC_ID'])['DEBT'].agg({'mean','median','std'})
students_mark_history_test = train[train['ST_YEAR'].isin(['2019','2020'])].groupby(by=['DISC_ID'])['DEBT'].agg({'mean','median','std'})
