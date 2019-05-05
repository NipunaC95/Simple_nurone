#numpy for matrix multiplication
import numpy as np
import time 

#variables 
n_hidden = 10
n_inputs = 10
n_outputs = 10

#sample data 
n_samples = 300

#hyper parameters 
learning_rate = 0.01 #Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient.
momentum = 0.9

#non deterministic seeding
np.random.seed(0)

#activation functions
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
    return 1 - np.tanh(x)**2



#training function 
    #x : input data
    #t :transpose 
    # V,W : 2 layers
    # bv ,bw : biases of layers
def train(x , t , V , W , bv , bw ):
    #forwerd propergration  -> matrix multiplication  + biases
    A = np.dot(x ,V) + bv
    Z = np.tanh(A)

    #Z is the output from first layer and input of second layer
    
    B = np.dot(Z,W) + bw
    Y = sigmoid(B)

    #backword propergation 
    Ew = Y -t
    Ev = tanh_prime(A) * np.dot(W ,Ew)

    #predict our lose 
    dW = np.outer(Z , Ew)
    dV = np.outer(x , Ev)

    #cross entrophy
    loss = -np.mean(t * np.log(Y) * (1-t) * np.log(1-Y))

    return loss, (dV , dW , Ev , Ew)


def predict(x , V , W , bv , bw ):
    A = np.dot(x , V) + bv
    B = np.dot( np.tanh(A) , W )  + bw
    return (sigmoid(B)  >  0.5).astype(int)

#creating layers
V = np.random.normal(scale = 0.1 , size=(n_inputs, n_hidden))
W = np.random.normal(scale = 0.1 , size=(n_hidden, n_outputs))

#biases 
bv = np.zeros(n_hidden)
bw = np.zeros(n_outputs)

parameters = [V, W , bv , bw]

#genarate data 
X = np.random.binomial(1 , 0.5 ,(n_samples , n_inputs))  
T = X ^ 1 

#training time
for epoch in range(100):
    err = []
    upd = [0]*len(parameters)

    t0  = time.process_time()
    #for each data point , update our weights 
    for  i in range(X.shape[0]):
        loss , grad  = train(X[i] , T[i] , *parameters)
    
        #update loss
        for j in range(len(parameters)):
            parameters[j] -= upd[j]
        
        for j in range(len(parameters)):
            upd[j] = learning_rate * grad[j] * momentum * upd[j]

        err.append(loss)
    
    print('Epoch :%d , loss : %d , time : %.4f s'%(epoch , np.mean(err) , time.process_time()-t0))


#test data and output
x = np.random.binomial(1 , 0.5 , n_inputs)
print('XOR prediction ')
print(x)
print(predict(x ,*parameters))  