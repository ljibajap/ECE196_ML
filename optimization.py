
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
        print("Number Training Iterations: " + str(number_iterations) + " - Validation Loss: " + str(round(min_validation, 4)))
    min_validation_idx = np.argmin(validation_loss_list)
    best_model = models[min_validation_idx]

    validation_dic = {'Number Iterations': tune_numitr_list, \
        'Validation Loss': validation_loss_list}
    val_table = pd.DataFrame.from_dict(validation_dic)
    return (best_model,val_table)