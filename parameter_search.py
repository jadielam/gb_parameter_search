'''
Created on Sep 26, 2016

@author: dearj019
'''

import sys
import json
from merchandising.streaming.producers import update_tags_readers_map
from merchandising.filters.feature_extraction import advanced_feature_extraction
from merchandising.streaming.consumers import parse_raw_message
import os
import xgboost as xgb
import numpy as np    

def single_model(training_matrix, validation_matrix):
    num_boost_round = 5000
    watchlist = [(training_matrix, 'train'), (validation_matrix, 'eval')]
    eval_metric = "rmse"
    print("Training model  ...")
    
    #1. Step 1: Fix learning rate and number of estimators for tuning tree-based parameters
    eta = 0.1
    gamma = 0
    subsample = 0.6
    colsample_bytree = 0.6
    max_depth = 5
    min_child_weight = 1
    
    params = {"objective": "reg:linear",
                        "booster": "gbtree",
                          "eval_metric": eval_metric,
                          "eta": eta,
                          "max_depth": max_depth,
                          "subsample": subsample,
                          "colsample_bytree": colsample_bytree,
                          "min_child_weights": min_child_weight,
                          "silent": 1,
                          "gamma": gamma,
                          "seed": 0
                        }
    
    pos_model = xgb.train(params, training_matrix, num_boost_round, evals = watchlist,
                     early_stopping_rounds = 50, verbose_eval = True)
    return pos_model
    
def xgb_parameter_search(training_matrix, validation_matrix):
    '''This function implements the advice found in an online blog regarding how to tune xgb parameters:
    
    https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    
    It takes as input the training and validation matrices, and returns the best built model
    after doing a parameter search.
    '''
    num_boost_round = 5000
    watchlist = [(training_matrix, 'train'), (validation_matrix, 'eval')]
    eval_metric = "rmse"
    print("Training model  ...")
    
    #1. Step 1: Fix learning rate and number of estimators for tuning tree-based parameters
    eta = 0.1
    min_child_weight = 1
    gamma = 0
    subsample = 0.6
    colsample_bytree = 0.6
    #scale_pos_weight = 1
    
    #2. Step 2: Tune max_depth and min_child_weight
    print("Tunning max_depth and min_child_weight")
    
    max_depth_l = range(3, 10, 2)
    min_child_weight_l = range(1, 6, 2)
    
    min_error = 50.0
    best_max_depth = max_depth_l[0]
    best_min_child_weight = min_child_weight_l[0]
    
    for max_depth in max_depth_l:
        for min_child_weight in min_child_weight_l:
            params = {"objective": "reg:linear",
                        "booster": "gbtree",
                          "eval_metric": eval_metric,
                          "eta": eta,
                          "max_depth": max_depth,
                          "subsample": subsample,
                          "colsample_bytree": colsample_bytree,
                          "min_child_weights": min_child_weight,
                          "silent": 1,
                          "gamma": gamma,
                          "seed": 0
                        }
    
            evals_result = {}
            pos_model = xgb.train(params, training_matrix, num_boost_round, evals = watchlist,
                     early_stopping_rounds = 50, verbose_eval = True, evals_result = evals_result)
            
            min_local_error = float(min(evals_result['eval'][eval_metric]))
            
            if min_local_error < min_error:
                min_error = min_local_error
                best_max_depth = max_depth
                best_min_child_weight = min_child_weight
    
    print("min_error :" +str(min_error))
    print("best_max_depth: " + str(best_max_depth))
    print("best_min_child_weight" + str(best_min_child_weight))
    
    #3. Step 3: Tune gamma:
    print("Tunning gamma")
    gamma_l = [i/10.0 for i in range(0, 5)]
    best_gamma = gamma
    
    for gamma in gamma_l:
        params = {"objective": "reg:linear",
                        "booster": "gbtree",
                          "eval_metric": eval_metric,
                          "eta": eta,
                          "max_depth": best_max_depth,
                          "subsample": subsample,
                          "colsample_bytree": colsample_bytree,
                          "min_child_weights": best_min_child_weight,
                          "silent": 1,
                          "gamma": gamma,
                          "seed": 0
                        }
        evals_result = {}
        
        pos_model = xgb.train(params, training_matrix, num_boost_round, evals = watchlist,
                     early_stopping_rounds = 50, verbose_eval = False, evals_result = evals_result)
            
        min_local_error = float(min(evals_result['eval'][eval_metric]))
            
        if min_local_error < min_error:
            min_error = min_local_error
            best_gamma = gamma
    
    print("min_error :" +str(min_error))
    print("best_gamma: " + str(best_gamma))
    
    #4. Step 4: Tuning subsample and colsample_bytree
    print("Tunning subsample and colsample_bytree")
    subsample_l = [i/10.0 for i in range(6, 10)]
    best_subsample = subsample
    
    colsample_bytree_l = [i/10.0 for i in range(6, 10)]
    best_colsample_bytree = colsample_bytree
    for subsample in subsample_l:
        for colsample_bytree in colsample_bytree_l:
            params = {"objective": "reg:linear",
                        "booster": "gbtree",
                          "eval_metric": eval_metric,
                          "eta": eta,
                          "max_depth": best_max_depth,
                          "subsample": subsample,
                          "colsample_bytree": colsample_bytree,
                          "min_child_weights": best_min_child_weight,
                          "silent": 1,
                          "gamma": best_gamma,
                          "seed": 0
                        }
            evals_result = {}
        
            pos_model = xgb.train(params, training_matrix, num_boost_round, evals = watchlist,
                     early_stopping_rounds = 50, verbose_eval = False, evals_result = evals_result)
            
            min_local_error = float(min(evals_result['eval'][eval_metric]))
            
            if min_local_error <= min_error:
                min_error = min_local_error
                best_subsample = subsample
                best_colsample_bytree = colsample_bytree
            
    print("min_error :" +str(min_error))
    print("best_subsample: " + str(best_subsample))
    print("best_colsample_bytree: " + str(best_colsample_bytree))
    
    #5. Step 5: Tuning regularization parameters
    print("Regularization parameters")
    reg_alpha_l = [1e-5, 1e-2, 0.1, 0, 1, 100]
    best_reg_alpha = reg_alpha_l[0]
    for reg_alpha in reg_alpha_l:
        params = {"objective": "reg:linear",
                    "booster": "gbtree",
                      "eval_metric": eval_metric,
                      "eta": eta,
                      "max_depth": best_max_depth,
                      "subsample": best_subsample,
                      "colsample_bytree": best_colsample_bytree,
                      "min_child_weights": best_min_child_weight,
                      "reg_alpha": reg_alpha,
                      "silent": 1,
                      "gamma": best_gamma,
                      "seed": 0
                    }
        evals_result = {}
    
        pos_model = xgb.train(params, training_matrix, num_boost_round, evals = watchlist,
                 early_stopping_rounds = 50, verbose_eval = False, evals_result = evals_result)
        
        min_local_error = float(min(evals_result['eval'][eval_metric]))
        
        if min_local_error < min_error:
            min_error = min_local_error
            best_reg_alpha = reg_alpha
            
    
    print("min_error :" +str(min_error))
    print("best_reg_alpha: " + str(best_reg_alpha))
    
    #6. Tune learning rate:
    print("Tunning learning rate")
    eta_l = [0.001, 0.01, 0.05, 0.1, 0.5]
    best_eta = eta
    for eta in eta_l:
        params = {"objective": "reg:linear",
                    "booster": "gbtree",
                      "eval_metric": eval_metric,
                      "eta": eta,
                      "max_depth": best_max_depth,
                      "subsample": best_subsample,
                      "colsample_bytree": best_colsample_bytree,
                      "min_child_weights": best_min_child_weight,
                      "reg_alpha": best_reg_alpha,
                      "silent": 1,
                      "gamma": best_gamma,
                      "seed": 0
                    }
        evals_result = {}
    
        pos_model = xgb.train(params, training_matrix, num_boost_round, evals = watchlist,
                 early_stopping_rounds = 50, verbose_eval = False, evals_result = evals_result)
        
        min_local_error = float(min(evals_result['eval'][eval_metric]))
        
        if min_local_error < min_error:
            min_error = min_local_error
            best_eta = eta
    
    print("min_error :" +str(min_error))
    print("best_eta: " + str(best_eta))
 
    #Finally: build the best model:
    params = {"objective": "reg:linear",
                    "booster": "gbtree",
                      "eval_metric": eval_metric,
                      "eta": best_eta,
                      "max_depth": best_max_depth,
                      "subsample": best_subsample,
                      "colsample_bytree": best_colsample_bytree,
                      "min_child_weights": best_min_child_weight,
                      "reg_alpha": best_reg_alpha,
                      "silent": 1,
                      "gamma": best_gamma,
                      "seed": 0
                    }
    
    pos_model = xgb.train(params, training_matrix, num_boost_round, evals = watchlist,
                 early_stopping_rounds = 50, verbose_eval = False, evals_result = evals_result)
    return pos_model
    
def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
        
    readers = conf['readers']
    model_output_folder = conf['model_output_folder']
    training_files = conf['file_paths']
    
    #1. Read from files
    queue = []
    for i in range(len(training_files)):
        if i > 0:
            break
        
        t_file = training_files[i]
        with open(t_file) as f:
            text = f.read()
            text = text.split("\n")
            
        #2. Throw into the queue the info from files (throw a marker as end of the queue)
        for line in text:
            if len(line) > 1:
                line_d = parse_raw_message(line)
                queue.append(line_d)
        
        queue.append(None)
        
    #3. From queue have a reader to organize into a map of Tags
    #4. Do feature extraction and accumulate those values.
    tags_readers_d = {}
    tag_counters_d = {}
    X_all = []
    Y1_all = []
    Y2_all = []
    
    print("Feature extraction...")
    total_count = 0
    error_count = 0
    print(len(queue))
    for i in range(len(queue)):
        if i % 10000 == 0:
            print(i)
        if i > 300000:
            break
        entries_d = queue[i]
        if entries_d is not None:
            update_tags_readers_map(entries_d, tags_readers_d, tag_counters_d)
            
            
            if i % 25 == 0:
                for tag in tags_readers_d.keys():
                    
                    tag_features = advanced_feature_extraction(readers, tags_readers_d, tag)
                    #tag_features = extract_features_from_raw_entries(readers, tags_readers_d, tag)
                    try:    
                        total_count = total_count + 1
                        x_pos = tags_readers_d[tag][readers[0]][-1]['x_pos']
                        y_pos = tags_readers_d[tag][readers[0]][-1]['y_pos']
                    
                        X_all.append(tag_features)
                        Y1_all.append(x_pos)
                        Y2_all.append(y_pos)
                    except:
                        error_count = error_count + 1
                        
                        #print("error here")
                        #print(total_count)
                        #print(error_count)
                        continue
            
    #4. Remove duplicates here and split into training and validation.
    X_training = []
    Y1_training = []
    Y2_training = []
    X_validation = []
    Y1_validation = []
    Y2_validation = []
    
    print("Building training and validation sets")
    '''
    for i in range(len(X_all)):
        if i % 5 == 0:
            X_validation.append(X_all[i])
            Y1_validation.append(Y1_all[i])
            Y2_validation.append(Y2_all[i])
        else:
            X_training.append(X_all[i])
            Y1_training.append(Y1_all[i])
            Y2_training.append(Y2_all[i])
    
    '''
    cutoff = int(len(X_all) / 5.0)
    X_training = X_all[:-cutoff]
    Y1_training = Y1_all[:-cutoff]
    Y2_training = Y2_all[:-cutoff]
    X_validation = X_all[-cutoff:]
    Y1_validation = Y1_all[-cutoff:]
    Y2_validation = Y2_all[-cutoff:]
            
    X_training = np.array(X_training)
    X_validation = np.array(X_validation)
    
    #5. Build model using xgb
    X_Y1_training = xgb.DMatrix(X_training, Y1_training, missing = np.NaN)
    X_Y2_training = xgb.DMatrix(X_training, Y2_training, missing = np.NaN)        
    X_Y1_validation = xgb.DMatrix(X_validation, Y1_validation, missing = np.NaN)
    X_Y2_validation = xgb.DMatrix(X_validation, Y2_validation, missing = np.NaN)
    
    print("Training model x ...")
    x_pos_model = xgb_parameter_search(X_Y1_training, X_Y1_validation)
    
    print("Training model y ...")
    y_pos_model = xgb_parameter_search(X_Y2_training, X_Y2_validation)
    
    x_pos_model.save_model(os.path.join(model_output_folder, "x_pos_model"))
    y_pos_model.save_model(os.path.join(model_output_folder, "y_pos_model"))

if __name__ == "__main__": 
    main()
