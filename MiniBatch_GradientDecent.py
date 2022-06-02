
import numpy as np 

def miniBatch_GradientDecent(data, labels, lr = 0.001, epochs = 100, batch_size = 4):

    #data = np.insert(np.ones(len(data)), data, axis=1)
    data = np.column_stack((np.ones(len(data)), data))
    params = np.zeros(data.shape[1])

    len_data = len(labels)
    data_per_batch = len_data // batch_size
    
    # these lists are to follow up the training process 
    list_params  = []
    list_cost    = []
    list_ypredicted  = []

    for i in range(epochs):
        for j in range (batch_size):
            batch_data = data[j*data_per_batch : data_per_batch*(j+1)]
            batch_labels = labels[j*data_per_batch : data_per_batch*(j+1)]

            h = np.dot(batch_data, params)
            error = h - batch_labels
            cost = np.dot(error, error) / (2 * len(batch_labels))
            gradient_params = np.sum(error*batch_data) / len(batch_data)
            params = params - lr*gradient_params

        y_predicted = data @ params

        list_params.append(params)
        list_cost.append(cost)
        list_ypredicted.append(y_predicted)
    
    return list_params, list_cost, list_ypredicted, params, y_predicted
    