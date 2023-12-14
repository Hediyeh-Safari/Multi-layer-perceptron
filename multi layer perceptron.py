import numpy as np
#import matplotlib.pyplot as plt #<---- for final submission we will comment this out

#Hidden Layer definition
class Hidden_Layer:
    def __init__(self, num_inputs,num_perceptrons):
        #initialize weights and biases base on the # inputs and perceptrons definied, tranpose w matrix
        self.w = 0.1 * np.random.randn(num_inputs,num_perceptrons)
        self.b = np.zeros((1,num_perceptrons))

    #hidden layer forward pass definition
    def forward(self,inputs):
        self.inputs = inputs
        #perform dot product on the input and weights eq = sum(w*x) + b
        self.out = np.dot(inputs,self.w) + self.b
        self.act_in = self.out
        #pass the peceptron output to the ReLU activiation function
        self.act_out = np.maximum(0,self.act_in)

    #hidden layer backward pass definition
    def backward(self,grad):
        #get gradiation from the activiation function
        self.act_grad = grad.copy()
        # calculate the derivative
        self.act_grad[self.act_in <= 0] = 0
        #calculate the delta change from gradient to update the weight and bias
        self.delta_w = np.dot(self.inputs.T,self.act_grad)
        self.delta_b = np.sum(self.act_grad,axis=0,keepdims=True)  
        self.delta_in = np.dot(self.act_grad,self.w.T) 

#output layer definition
class Output_Layer:
    
    def __init__(self, num_inputs,num_perceptrons):
        #initialize weights and biases base on the # inputs and perceptrons definied, tranpose w matrix
        self.w = 0.1 * np.random.randn(num_inputs,num_perceptrons)
        self.b = np.zeros((1,num_perceptrons))

    #output layer forward pass definition
    def forward(self,inputs):
        #pass output from hidden layer as input to this layer
        self.inputs = inputs
        #perform dot product on all if the inputs
        self.out = np.dot(inputs,self.w) + self.b
        self.act_in = self.out
        #pass output of perceptron to the softmax activation function
        act_exp = np.exp(self.act_in - np.max(self.act_in,axis=1,keepdims=True))
        act_prob =  act_exp/np.sum(act_exp,axis=1,keepdims=True)
        #save softmax activation output
        self.act_out = act_prob

    
    def backward(self,grad):
        #calculate delta change from gradient
        self.delta_w = np.dot(self.inputs.T,grad)
        self.delta_b = np.sum(grad,axis=0,keepdims=True)  
        self.delta_in = np.dot(grad,self.w.T)  

#loss function definition
class Loss_Function:
    # calculate loss
    def forward(self,predictions,actuals):
        samples = len(predictions)
        predictions_c = np.clip(predictions,1e-7, 1-1e-7)

        if len(actuals.shape) == 1:
            pred_confi = predictions_c[range(samples),actuals]
        elif len(actuals.shape) == 2:
            pred_confi = np.sum(predictions_c*actuals,axis=1)
        
       
        loss = np.mean(-np.log(pred_confi))
        return loss
    
    #calculate gradient base on the prediction output
    def backward(self, prediction, actual):
        samples = len(prediction)

        if len(actual.shape) == 2:
            actual = np.argmax(actual, axis=1)
        
        self.delta_in = prediction.copy()
        #calculate gradient with cross categorical entropy for one-hot encoded outpput
        self.delta_in[range(samples), actual] -= 1
        #normalize the data
        self.delta_in = self.delta_in / samples

# function to update the weight and biases base on the delta calculated
def update(layer,learning_rate):
    layer.w += -learning_rate * layer.delta_w
    layer.b += -learning_rate * layer.delta_b

# function to train the neural net
def train_nn(train_data,train_labels):

    #load training data
    training = np.loadtxt(train_data,delimiter=',').astype(float)
    label = np.loadtxt(train_labels,delimiter=',').astype(float)

    if len(training.shape) == 1:
        training = [training]
    if len(label.shape) == 1:
        label = [label]

    #initilize the neural net layers
    hidden_layer = Hidden_Layer(784,100)
    output_layer = Output_Layer(100,4)
    loss_function = Loss_Function()

    accuracy_arr = []
    loss_arr = []
  
    #training loop, set epoch here
    for epoch in range(500):
        #define the batches size to be train and take the first 20000 from training data to be our data set
        epoch_accuracy = []
        epoch_loss = []
        

        #loop through training set at a step of 50 each iteration
        for index in range(0,len(training),50):

            #split training set into batches of 50
            if ((len(training)-(index)) < 50):
                training_batch = np.array(training[index:len(training)])
                label_batch = np.array(label[index:len(training)])
            else:
                training_batch = np.array(training[index:index+50])
                label_batch = np.array(label[index:index+50])
            
            if len(training_batch.shape) == 1:
                training_batch = [training_batch]
            if len(label_batch.shape) == 1:
                label_batch = [label_batch]

            # training_batch = training[(index-50):index]
            # label_batch = label[(index-50):index]
    
            hidden_layer.forward(training_batch)
            output_layer.forward(hidden_layer.act_out)
            #calculate loss
            loss = loss_function.forward(output_layer.out, label_batch)
            epoch_loss.append(loss)
            

            predictions = np.argmax(output_layer.act_out, axis=1)

            if len(label_batch.shape) == 2:
                label_batch = np.argmax(label_batch, axis=1)

            #calculate accuracy
            accuracy = np.mean(predictions==label_batch)
            epoch_accuracy.append(accuracy)
   
            #perform backward pass
            loss_function.backward(output_layer.act_out, label_batch)
            output_layer.backward(loss_function.delta_in)
            hidden_layer.backward(output_layer.delta_in)
            
            #update the weight and biases base on delta calculated during backward pass
            update(output_layer,0.01)
            update(hidden_layer,0.01)

        #output the average accuracy and loss for the training set per epoch
        print('epoch:',epoch,'accuracy:','%.5f' %np.mean(epoch_accuracy),'loss:','%.5f' %np.mean(epoch_loss))
        accuracy_arr.append(np.mean(epoch_accuracy))
        loss_arr.append(np.mean(epoch_loss))

    
    #---------------CODE USE TO SAVE WEIGHT AND BIASES OF NEURAL NET IN CSV FILE, for final submission we will comment these out---------------------#

    # np.savetxt('hidden_layer_weights.csv',hidden_layer.w,delimiter=',')
    # np.savetxt('hidden_layer_bias.csv',hidden_layer.b,delimiter=',')
    # np.savetxt('output_layer_weights.csv',output_layer.w,delimiter=',')
    # np.savetxt('output_layer_bias.csv',output_layer.b,delimiter=',')

    #----------------CODE USE TO PLOT ACCURACY AND LOSS GRAPH, for final submission we will comment these out----------------------------#
   
    # accuracy_plot = plt
    # accuracy_plot.title("Accruacy vs # Epoch")
    # accuracy_plot.xlabel("Epoch")
    # accuracy_plot.ylabel("Accuracy")
    # accuracy_plot.plot(accuracy_arr)
    # accuracy_plot.savefig('plot_accuracy.pdf')
    # plt.clf()

    # loss_plot = plt
    # loss_plot.title("Loss vs # Epoch")
    # loss_plot.xlabel("Epoch")
    # loss_plot.ylabel("Loss")
    # loss_plot.plot(loss_arr)
    # loss_plot.savefig('plot_loss.pdf')
 
   
#test function
def test_nn(test_data):

    #load test data
    inputs = np.loadtxt(test_data,delimiter=',').astype(float)
  
    #load weight and biases
    hidden_layer_weights = np.loadtxt('hidden_layer_weights.csv',delimiter=',').astype(float)
    hidden_layer_bias = np.loadtxt('hidden_layer_bias.csv',delimiter=',').astype(float)
    output_layer_weights = np.loadtxt('output_layer_weights.csv',delimiter=',').astype(float)
    output_layer_bias = np.loadtxt('output_layer_bias.csv',delimiter=',').astype(float)

    #initialize neural net
    hidden_layer = Hidden_Layer(784,100)
    output_layer = Output_Layer(100,4)

    #load weight and biases into neural net
    hidden_layer.w = hidden_layer_weights
    output_layer.b = hidden_layer_bias
    output_layer.w = output_layer_weights
    output_layer.b = output_layer_bias


    #perform forward pass
    hidden_layer.forward(inputs)
    output_layer.forward(hidden_layer.act_out)

    #save predication
    prediction = np.round(output_layer.act_out)

    #return prediction
    return prediction
        


#------------CODE USE TO TRAIN THE NEURAL NET------------------#

#train_nn('train_data.csv','train_labels.csv')


#------------CODE USE TO TEST THE NEURAL NET------------------#

# prediction = test_nn('test_data.csv')
# actual = np.loadtxt('test_labels.csv',delimiter=',').astype(float)
# accuracy = np.mean(prediction==actual)

# print(prediction)
# print('accuracy:', accuracy)



