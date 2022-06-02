
import numpy as np

def batch_gradientDecent(data, labels, lr = 0.001, epochs= 100):
    
    data = np.column_stack((np.ones(len(data)), data))
    params = np.zeros(data.shape[1])
    
    # these lists are to follow up the training process 
    list_params  = []
    list_cost    = []
    list_ypredicted  = []

    for i in range(epochs):
        h = np.dot(data, params)
        error = h - labels
        cost = np.dot(error, error) / (2 * len(labels))
        gradient_params = np.dot(data.T, error) / len(labels)
        params = params - lr*gradient_params

        # append result of each epoch
        list_params.append(params)
        list_cost.append(cost)
        list_ypredicted.append(h)
    
    y_predicted = data @ params
    
    return list_params, list_cost, list_ypredicted, params, y_predicted