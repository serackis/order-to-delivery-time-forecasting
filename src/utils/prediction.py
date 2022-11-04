# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Utility functions for inference
'''

import time
import numpy as np
from sklearn.metrics import accuracy_score


def xgb_predict_reg(xgb_model, x_test_reg, y_test_reg):
    xgb_pred_start = time.time()
    y_pred = xgb_model.predict(x_test_reg)
    xgb_pred_time = time.time() - xgb_pred_start
    xgb_mse = np.square(np.subtract(y_test_reg.values.reshape(-1), y_pred)).mean()
    return xgb_mse, xgb_pred_time
        
def rf_predict_reg(rf_model, x_test_reg, y_test_reg):
    rf_pred_start = time.time()
    y_pred = rf_model.predict(x_test_reg)
    rf_pred_time = time.time() - rf_pred_start
    rf_mse = np.square(np.subtract(y_test_reg.values.reshape(-1), y_pred)).mean()
    return rf_mse, rf_pred_time

def SV_predict_reg(svr_model, x_test_reg, y_test_reg):
    svr_pred_start = time.time()
    y_pred = svr_model.predict(x_test_reg)
    svr_pred_time = time.time() - svr_pred_start
    svr_mse = np.square(np.subtract(y_test_reg.values.reshape(-1), y_pred)).mean()
    return svr_mse, svr_pred_time

def ensemble_predict_reg(ensemble_model, x_test_reg, y_test_reg):
    ensemble_pred_start = time.time()
    y_pred = ensemble_model.predict(x_test_reg)
    ensemble_pred_time = time.time() - ensemble_pred_start
    ensemble_mse = np.square(np.subtract(y_test_reg.values.reshape(-1), y_pred)).mean()
    return ensemble_mse, ensemble_pred_time

def xgb_predict_class(xgb_model, x_test_class, y_test_class):
    xgb_pred_start = time.time()
    y_pred = xgb_model.predict(x_test_class)
    xgb_pred_time = time.time() - xgb_pred_start
    xgb_acc = accuracy_score(y_test_class, y_pred)
    return xgb_acc, xgb_pred_time
        
def rf_predict_class(rf_model, x_test_class, y_test_class):
    rf_pred_start = time.time()
    y_pred = rf_model.predict(x_test_class)
    rf_pred_time = time.time() - rf_pred_start
    rf_acc = accuracy_score(y_test_class, y_pred)
    return rf_acc, rf_pred_time

def SV_predict_class(svc_model, x_test_class, y_test_class):
    svc_pred_start = time.time()
    y_pred = svc_model.predict(x_test_class)
    svc_pred_time = time.time() - svc_pred_start
    svc_acc = accuracy_score(y_test_class, y_pred)
    return svc_acc, svc_pred_time

def ensemble_predict_class(ensemble_model, x_test_class, y_test_class):
    ensemble_pred_start = time.time()
    y_pred = ensemble_model.predict(x_test_class)
    ensemble_pred_time = time.time() - ensemble_pred_start
    ensemble_acc = accuracy_score(y_test_class, y_pred)
    return ensemble_acc, ensemble_pred_time
