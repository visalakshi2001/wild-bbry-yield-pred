import joblib
import matplotlib as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
import shap
from sklearn.ensemble import RandomForestRegressor
import sklearn

curr_path = os.path.dirname(os.path.realpath(__file__))

feat_cols = ['clonesize', 'osmia', 'AverageOfUpperTRange', 'AverageOfLowerTRange', 
       'AverageRainingDays', 'fruitset', 'fruitmass', 'seeds']

X_test_rf_df = pd.read_csv(curr_path + "/assets/X_test_rf_df.csv", index_col= 0)
rf_final = joblib.load(curr_path + "/assets/joblib_files/rf_bbry_tuned_model.pkl")

def predict_yield(attributes: np.ndarray):
    """ Returns Blueberry Yield value"""
    # print(attributes.shape) # (1,8)

    shap_explainer = shap.TreeExplainer(rf_final)
    shap_values = shap_explainer.shap_values(attributes)
    shap_expected_values = shap_explainer.expected_value

    # plt.figure(figsize=(9,13))
    shap.force_plot(shap_expected_values, 
                    shap_values, 
                    attributes, 
                    feat_cols, 
                    show=False, 
                    matplotlib=True).savefig(curr_path + "/assets/force_plot_custom.png",
                                             bbox_inches = 'tight')

    image = Image.open(curr_path + '/assets/force_plot_custom.png')


    pred = rf_final.predict(attributes)
    print("Yield predicted")

    return pred[0], image