# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Utility functions for data processing
'''

from math import radians, sin, cos, asin, sqrt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def read_data():
    # Read CSV files with proper datetime parsing
    orders = pd.read_csv("./data/olist_orders_dataset.csv", 
                        parse_dates=['order_purchase_timestamp', 'order_approved_at', 
                                   'order_delivered_carrier_date', 'order_delivered_customer_date', 
                                   'order_estimated_delivery_date'])
    items = pd.read_csv("./data/olist_order_items_dataset.csv")
    customers = pd.read_csv("./data/olist_customers_dataset.csv")
    sellers = pd.read_csv("./data/olist_sellers_dataset.csv")
    geo = pd.read_csv("./data/olist_geolocation_dataset.csv")
    products = pd.read_csv("./data/olist_products_dataset.csv")
    return orders, items, customers, sellers, geo, products


def preprocessing(orders, items, customers, sellers, geo, i_flag):
    # Merge/Clean Data

    # Get the seller zip code of each order
    middle = items[['order_id', 'seller_id']]
    middle_2 = middle.merge(sellers[['seller_id', 'seller_zip_code_prefix']], on="seller_id", how="outer")
    orders = orders.merge(middle_2, on="order_id", how="left")

    # Get customer zip code of each order
    orders = orders.merge(customers[['customer_id', 'customer_zip_code_prefix']],
                          on='customer_id', how="left")

    # Clean geo df
    geo = geo[~geo['geolocation_zip_code_prefix'].duplicated()]

    # add seller coordinates to the orders
    orders = orders.merge(geo, left_on="seller_zip_code_prefix",
                          right_on="geolocation_zip_code_prefix", how="left")

    # add customer coordinates to the orders
    orders = orders.merge(geo, left_on="customer_zip_code_prefix",
                          right_on="geolocation_zip_code_prefix", how="left",
                          suffixes=("_seller", "_customer"))
    # Clean orders
    # 1-Filter out orders with multiple sellers Because each order only has one delivery date
    df = orders.groupby(by="order_id").nunique()
    mono_orders = pd.Series(df[df['seller_id'] == 1].index)
    filtered_orders = orders.merge(mono_orders, how='inner')

    # 2-drop rows with missing values
    filtered_orders = filtered_orders.drop(columns=["order_approved_at"])
    filtered_orders = filtered_orders.dropna()
    return filtered_orders


# Define Function to calculate distance
def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Compute distance between two pairs of (lat, lng)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def get_package_size(items, products):
    # Get Package Size
    df_tmp = items[['order_id', 'product_id']].merge(products[['product_id', 'product_length_cm', 'product_height_cm',
                                                               'product_width_cm', 'product_weight_g']],
                                                     on="product_id",
                                                     how="outer")
    df_tmp.loc[:, "product_size_cm3"] = \
        df_tmp['product_length_cm']*df_tmp['product_width_cm'] * df_tmp['product_height_cm']
    orders_size_weight = df_tmp.groupby("order_id", as_index=False).sum()[['order_id', 'product_size_cm3',
                                                                           'product_weight_g']]
    return orders_size_weight

def object_to_int(dataframe_series):

    if dataframe_series.dtype == 'object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series


def split_data(final_df, columns_for_train, columns_for_pred, i_flag):

    if i_flag:
        from sklearnex import patch_sklearn  # pylint: disable=C0415, E0401
        patch_sklearn()
    from sklearn.model_selection import train_test_split  # pylint: disable=C0415

    x_train, x_test, y_train, y_test = train_test_split(final_df[columns_for_train], final_df[columns_for_pred],
                                                        test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test
    
def scaling(X_train, X_test):
    sc_X = StandardScaler()
    X_train_scaled = sc_X.fit_transform(X_train)
    X_test_scaled = sc_X.fit_transform(X_test)
    return X_train_scaled, X_test_scaled
