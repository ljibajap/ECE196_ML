
import pandas as pd
import numpy as np
import model as md

# Tunning the number of iterations experiment function
def tunning_number_iterations(train_data,val_data,list_userId,list_movieId,num_tunning=10):
    tune_numitr_list =  np.zeros(num_tunning,dtype=np.int)
    validation_loss_list = []
    models = []
    for i in range(1,num_tunning + 1,1):
        number_iterations = i * num_tunning
        tune_numitr_list[i-1] = number_iterations
        model = md.CF_Model(list_userId, list_movieId)
        model.build_model(num_features=5,std=1)
        model.train(train_data,val_data,num_iter=number_iterations,learning_rate=0.00001,regulator=0.001)
        min_validation = model.update_best_model()
        validation_loss_list.append(min_validation)
        models.append(model)
        print("Number Training Iterations: " + str(number_iterations) + " - Validation Loss: " + str(round(min_validation, 7)))
    min_validation_idx = np.argmin(validation_loss_list)
    best_model = models[min_validation_idx]

    validation_dic = {'Number Iterations': tune_numitr_list, \
        'Validation Loss': validation_loss_list}
    val_table = pd.DataFrame.from_dict(validation_dic)
    return (best_model,val_table)

'''
K-fold data break
'''
def k_foldCV(data_set, k=10):
    N = data_set.shape[0]
    size = int(N / k)
    r = N % k
    folds = []
    index = 0
    for i in range(k):
        if r > 0:
            folds.append(data_set[index:(index+size + 1)])
            r = r - 1
            index = index + size + 1
        else:
            folds.append(data_set[index:(index+size)])
            index = index + size
    return folds

# K-Cross Validation
# Return the best model from the K fold
def k_cross_validation(train_data, val_data, list_userId, list_movieId):
    train_folds = k_foldCV(train_data, k=5)
    val_folds = k_foldCV(val_data,k=5)
    val_loss_list = []
    models = []
    for fold in range (len(train_folds)):
        train_holdout = train_folds[fold]
        val_holdout = val_folds[fold]
        model = md.CF_Model(list_userId,list_movieId)
        model.build_model(num_features=5,std=1)
        model.train(train_holdout,val_holdout,num_iter=70,learning_rate=0.00001,            regulator=0.001)
        min_validation = model.update_best_model()
        val_loss_list.append(min_validation)
        models.append(model)
        print("Fold: " + str(fold) + " - Val Loss: " + str(round(min_validation,6)))
    
    min_validation_idx = np.argmin(val_loss_list)
    k_best_model = models[min_validation_idx]
    return k_best_model