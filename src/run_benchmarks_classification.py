# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Code for executing classification pipeline
'''

# pylint: disable=consider-using-with

import argparse
import logging
import pathlib
import time
import pickle  # nosec
import pandas as pd
import xgboost as xgb
from utils.data_functions import read_data, preprocessing, haversine_distance, get_package_size
from utils.data_functions import object_to_int
from utils.prediction import xgb_predict_class, rf_predict_class, SV_predict_class, ensemble_predict_class

def main(FLAGS):
    '''
    ###############
    # Data Read and Preprocessing start
    ###############
    '''

    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)

    logger = logging.getLogger()

    i_flag = FLAGS.intel
    if i_flag:
        from sklearnex import patch_sklearn  # pylint: disable=C0415, E0401
        patch_sklearn()
    
    from sklearn.model_selection import train_test_split  # pylint: disable=C0415
    from sklearn.model_selection import GridSearchCV  # pylint: disable=C0415
    from sklearn.ensemble import RandomForestClassifier  # pylint: disable=C0415
    from sklearn.ensemble import VotingClassifier  # pylint: disable=C0415
    from sklearn.svm import SVC  # pylint: disable=C0415
    
    logger.info("\n")
    if FLAGS.intel:
        logger.info("===== Running benchmarks for oneAPI tech =====")
    else:
        logger.info("===== Running benchmarks for stock =====")

    orders, items, customers, sellers, geo, products = read_data()
    logger.info("data read done. moving to preprocessing")
    filtered_orders = preprocessing(orders, items, customers, sellers, geo, i_flag)
    filtered_orders['distance'] = filtered_orders.apply(lambda row: haversine_distance(row["geolocation_lng_seller"],
                                                                                       row["geolocation_lat_seller"],
                                                                                       row["geolocation_lng_customer"],
                                                                                       row["geolocation_lat_customer"],),
                                                        axis=1,)

    orders_size_weight = get_package_size(items, products)
    filtered_orders = filtered_orders.merge(orders_size_weight, on='order_id', how='left')

    # Time columns are already parsed as datetime in read_data()
    # No need to convert them again

    filtered_orders.loc[:, "wait_time"] = (filtered_orders['order_delivered_customer_date'] -
                                           filtered_orders['order_purchase_timestamp']).dt.days
    filtered_orders.loc[:, "est_wait_time"] = (filtered_orders['order_estimated_delivery_date'] -
                                               filtered_orders['order_purchase_timestamp']).dt.days

    filtered_orders.loc[:, "purchase_dow"] = filtered_orders.order_purchase_timestamp.dt.dayofweek
    filtered_orders.loc[:, "year"] = filtered_orders.order_purchase_timestamp.dt.year
    filtered_orders.loc[:, "purchase_month"] = filtered_orders.order_purchase_timestamp.dt.month

    final_df = filtered_orders[['purchase_dow', 'purchase_month', 'year', 'product_size_cm3', 'product_weight_g',
                                'geolocation_state_customer', 'geolocation_state_seller', 'distance',
                                'wait_time', 'est_wait_time']]
    final_df['delay'] = final_df['wait_time'] - final_df['est_wait_time']
    final_df['delay'] = final_df['delay'] > 0
    final_df['delay'] = final_df['delay'].astype(int)

    final_df_enc = final_df.apply(lambda x: object_to_int(x))  # pylint: disable=W0108
    logger.info("preprocessing done")

    #
    # Data Read and Preprocessing end
    #

    # split data
    columns_for_classification_train = ['purchase_dow', 'purchase_month', 'year', 'product_size_cm3', 'product_weight_g',
                                        'geolocation_state_customer', 'geolocation_state_seller', 'distance',
                                        'est_wait_time']
    columns_for_classification_pred = ['delay']
    
    X_train_class_0, X_test_class_0, y_train_class_0, y_test_class_0 = train_test_split(final_df_enc[columns_for_classification_train],
                                                                                        final_df_enc[columns_for_classification_pred],
                                                                                        random_state=42)

    #

    # Hyperparameter Tuning for individual models (XGB, RF, SV)

    #
    
    hyper_train = pd.DataFrame(X_train_class_0, columns=columns_for_classification_train)
    hyper_train['delay'] = y_train_class_0['delay'].values
    hyper_train_section = hyper_train.sample(8192, random_state=42)
    X_train_class = hyper_train_section[columns_for_classification_train]
    y_train_class = hyper_train_section[columns_for_classification_pred]

    # Set parameters for xgb and random forests and SV for hyper parameter tuning
    xgb_params = {'n_estimators': [500, 1000], 'max_depth': [10, 20], 'tree_method': ['hist']}
    rf_params = {'n_estimators': [500, 1000], 'max_depth': [10, 20]}
    SV_params = {'C': [10, 20], 'kernel': ['rbf'], 'gamma': ['auto']}
    ensemble_params = {'voting': ['hard', 'soft'], 'weights': [(1, 1, 1), (1, 2, 1), (2, 1, 1), (1, 1, 2)]}
    
    logger.info('Running Hyperparameter Tuning for XGB')
    xgb_model = xgb.XGBClassifier(eval_metric='error', use_label_encoder=False)
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=4, n_jobs=-1, verbose=True)
    
    # record time
    xgb_start = time.time()
    xgb_grid.fit(X_train_class, y_train_class.values.ravel())
    xgb_hyper_time = time.time()-xgb_start
    
    # extract best grid
    xgb_best_grid = xgb_grid.best_estimator_

    logger.info('Hyperparameter tuning time for for: XGB: %s', str(xgb_hyper_time))
    logger.info('\n')

    logger.info('Running Hyperparameter Tuning for RF')
    rf_model = RandomForestClassifier()
    rf_grid = GridSearchCV(rf_model, rf_params, cv=4, n_jobs=-1, verbose=True)
    
    # record time
    rf_start = time.time()
    rf_grid.fit(X_train_class, y_train_class.values.ravel())
    rf_hyper_time = time.time()-rf_start
    
    # extract best grid and params
    rf_best_grid = rf_grid.best_estimator_

    logger.info('Hyperparameter tuning time for for: RF: %s', str(rf_hyper_time))
    logger.info('\n')
    
    logger.info('Running Hyperparameter Tuning for SV')
    SV_model = SVC(probability=True)
    SV_grid = GridSearchCV(SV_model, SV_params, cv=4, n_jobs=-1, verbose=True)

    # record time
    SV_start = time.time()
    SV_grid.fit(X_train_class, y_train_class.values.ravel())
    SV_hyper_time = time.time()-SV_start
    
    # extract best grid and params
    SV_best_grid = SV_grid.best_estimator_

    logger.info('Hyperparameter tuning time for for: SV: %s', str(SV_hyper_time))
    logger.info('\n')

    #

    # Hyperparameter Tuning for ensemble model using the "Best Grid for SGB, RF, SV"

    #

    logger.info('Running Hyperparameter Tuning for ensemble')
    voting_model = VotingClassifier(estimators=[('xgb', xgb_best_grid), ('rf', rf_best_grid), ('SV', SV_best_grid)])
    ensemble_grid = GridSearchCV(voting_model, ensemble_params, cv=4, n_jobs=-1, verbose=True)

    # record time
    ensemble_start = time.time()
    ensemble_grid.fit(X_train_class, y_train_class.values.ravel())
    ensemble_hyper_time = time.time()-ensemble_start
    
    # extract best params
    ensemble_best_params = ensemble_grid.best_params_

    logger.info('Hyperparameter tuning time for for: ensemble model: %s', str(ensemble_hyper_time))
    logger.info('\n')

    #

    # Extract Simple Training Benchmarks using best parameters for ensemble

    #
    
    ti_train = pd.DataFrame(X_train_class_0, columns=columns_for_classification_train)
    ti_train['delay'] = y_train_class_0['delay'].values

    for data_size in [8192, 32768]:
        ti_train_section = ti_train.sample(data_size, random_state=42)
        X_train_class = ti_train_section[columns_for_classification_train]
        y_train_class = ti_train_section[columns_for_classification_pred]

        logger.info('Running Training for ensemble model with data length %s', str(data_size))
        
        voting_model = VotingClassifier(estimators=[('xgb', xgb_best_grid), ('rf', rf_best_grid), ('SV', SV_best_grid)],
                                        n_jobs=-1, weights=ensemble_best_params['weights'], voting=ensemble_best_params['voting'])
        ensemble_model_train_start = time.time()
        voting_model.fit(X_train_class, y_train_class.values.ravel())
        ensemble_model_train_time = time.time() - ensemble_model_train_start
        
        logger.info('Training time for for: ensemble model: %s', str(ensemble_model_train_time))
        logger.info('\n')

    ti_test_base = pd.DataFrame(X_test_class_0, columns=columns_for_classification_train)
    ti_test_base['delay'] = y_test_class_0['delay'].values
    ti_test = pd.concat([ti_test_base, ti_test_base, ti_test_base, ti_test_base])
    
    for data_size in [10000, 30000, 50000, 70000]:
        ti_test_section = ti_test.sample(data_size, random_state=42)
        X_test_class = ti_test_section[columns_for_classification_train]
        y_test_class = ti_test_section[columns_for_classification_pred]
        
        # Perform prediction - refer to ensemble train function in utils
        xgb_mse, xgb_pred_time = xgb_predict_class(xgb_best_grid, X_test_class, y_test_class)
        rf_mse, rf_pred_time = rf_predict_class(rf_best_grid, X_test_class, y_test_class)
        SV_mse, SV_pred_time = SV_predict_class(SV_best_grid, X_test_class, y_test_class)
        ensemble_mse, ensemble_pred_time = ensemble_predict_class(voting_model, X_test_class, y_test_class)

        logger.info('Running Inference with data length %s', str(data_size))
        logger.info("Inference time and MSE for for XGB: %s", ' '.join(map(str, list((xgb_pred_time, xgb_mse)))))
        logger.info("Inference time and MSE for for RF: %s", ' '.join(map(str, list((rf_pred_time, rf_mse)))))
        logger.info("Inference time and MSE for for SV: %s", ' '.join(map(str, list((SV_pred_time, SV_mse)))))
        logger.info("Inference time and MSE for for Voting model: %s", ' '.join(map(str, list((ensemble_pred_time, ensemble_mse)))))
        logger.info('\n')
    
    pickle.dump(voting_model, open(FLAGS.modelfile, 'wb'))  # nosec
    voting_model = pickle.load(open(FLAGS.modelfile, 'rb'))  # nosec
    ti_test_section = ti_test.sample(1000, random_state=42)
    lst_of_stream_times = []
    for _counter in range(1000):
        sample_df = ti_test_section.sample(n=1)
        X_test_reg = sample_df[columns_for_classification_train]
        y_test_reg = sample_df[columns_for_classification_pred]
        ensemble_acc, ensemble_pred_time = ensemble_predict_class(voting_model, X_test_reg, y_test_reg)  # pylint: disable=W0612
        lst_of_stream_times.append(ensemble_pred_time)
    avg_stream_time = sum(lst_of_stream_times)/len(lst_of_stream_times)
    logger.info("Average Streaming Inference Time for Voting Classifier model: %s", str(avg_stream_time))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")

    parser.add_argument('-m_out',
                        '--modelfile',
                        type=str,
                        default="",
                        required=False,
                        help="dump model file after hyperparameter tuning")

    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        action="store_true",
                        help="use intel accelerated technologies")

    FLAGS = parser.parse_args()

    pd.options.mode.chained_assignment = None  # default='warn'
    
    main(FLAGS)
