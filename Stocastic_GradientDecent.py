import numpy as np

def stocastic_gradientDecent(data, labels, lr = 0.001, epochs = 100):
    
    data = np.column_stack((np.ones(len(data)), data))
    params = np.zeros(data.shape[1])
    len_data = len(labels)

    # these lists are to follow up the training process 
    list_params      = []
    list_cost        = []
    list_ypredicted  = []
    
    for i in range(epochs):
        for j in range (len_data):

            h = np.dot(data[j], params)
            error = h - labels[j]
            cost = np.dot(error, error) / 2 
            gradient_params = error*data[j]
            params = params - lr*gradient_params

        y_predicted = data @ params

        list_params.append(params)
        list_cost.append(cost)
        list_ypredicted.append(y_predicted)
    
    
    return list_params, list_cost, list_ypredicted, params, y_predicted
    