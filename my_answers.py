import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))


        
        
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes,self.output_nodes))
        
        
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x: 1/(1+np.exp(-x)) # Replace 0 with your sigmoid calculation.
        
        self.gradient_activation_function = lambda x : self.activation_function(x) * (1 - self.activation_function(x))
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 1/(1+np.exp(-x))
        
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        #print("features type ", type(features))
        #print ("target type", type (targets))

    
        n_records , n_features = features.shape

        
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        
        for X, y in zip(features, targets):
            X = X[:,None]
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
       
        
        #print ("features type, shape" , type (X), X.shape)
        #print ("self.weights_input_to_hidden type, shape" , type (self.weights_input_to_hidden), self.weights_input_to_hidden.shape)
        
        hidden_inputs = np.dot (X.T,self.weights_input_to_hidden) # signals into hidden layer
        #print ("hidden_inputs type, shape ", type(hidden_inputs), hidden_inputs.shape)
        #print (hidden_inputs)
        
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        #print ("hidden_outputs type, shape ", type(hidden_outputs), hidden_outputs.shape)
                      
        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot (hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        #print ("final_inputs type, shape ", type(final_inputs), final_inputs.shape)
        
        final_outputs = final_inputs # signals from final output layer

        

        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        #final_outputs = final_outputs[:, None]
        #hidden_outputs = hidden_outputs[:,None]
        #X = X[:, None]
        #y = y[:, None]
        #delta_weights_h_o = delta_weights_h_o.T
        '''
        print ("final output shape and type ", final_outputs.shape, type(final_outputs))
        print ("hidden outputs shape and type ", hidden_outputs.shape, type(hidden_outputs))
        print ("x shape and type", X.shape, type (X))
        print ("y shape and type", y, type (y))
        print (" w_i_h shape and type ",delta_weights_i_h.shape, type (delta_weights_i_h))
        print ("w_h_o shape and type ", delta_weights_h_o.shape, type (delta_weights_h_o))
        print (" ")
        print ("-----")
        print ("-----")
        print ("-----")
        '''
        
        # TODO: Output error - Replace this value with your calculations.
        # this error is the error of the output unit.
      
        
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        #The gradient of f(x) is 1. 
        output_grad = 1
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error * output_grad
        
        #print ("output_error_term shape ", output_error_term.shape)
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = output_error_term[0] * self.weights_hidden_to_output.T
        
        hidden_layer_grad = hidden_outputs * (1 - hidden_outputs)
      
  
        hidden_error_term = hidden_error * hidden_layer_grad
        
        #print("hidden error term shape", hidden_error_term.shape)
       
        #print ("the ugly shape", (output_error_term * hidden_outputs).shape) 
        # Weight step (hidden to output)
        delta_weights_h_o += (output_error_term * hidden_outputs).T
  
        
        # Weight step (input to hidden)
        delta_weights_i_h += X * hidden_error_term
                      
        return delta_weights_i_h, delta_weights_h_o
    

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        '''
        print ( "self.weights_hidden_to_output shape ", self.weights_hidden_to_output.shape)
        print ( "average delta_weights_h_o shape ", self.weights_hidden_to_output.shape)
        print ( "ugly shape 2", (self.lr * delta_weights_h_o/n_records ).T.shape)
        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o/n_records
        # update hidden-to-output weights with gradient descent step
        
        self.weights_input_to_hidden += self.lr * delta_weights_i_h/n_records  
        # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features,self.weights_input_to_hidden)# signals into hidden layer
        #print ( "hidden inputs ", hidden_inputs)
        
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        #print ( "hidden outputs ", hidden_outputs)
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)# signals into final output layer

        #print ( "final inputs ", final_inputs)

        final_outputs = final_inputs # signals from final output layer 
        
        #print ("actual output,", final_outputs[0,0])
        
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 9000
learning_rate = 0.25
hidden_nodes = 15
output_nodes = 1