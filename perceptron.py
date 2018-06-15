
import csv
import random
import numpy as np
import matplotlib.pyplot as plt


# READ CSV OF TRAINING
with open('iris_train.csv', 'r') as f:
    reader = csv.reader(f)
    training = list(reader)

training.pop(0)       # remove the first line of csv (sepalo,petalo,clase,especie)

# VARIABLES
to_learn = False      # boolean of learning
alpha    = 0.01       # speed of learning
errors   = []         # list of finding errors
plot_errors = []      # list of errors to plot
v_weight = [-1,2,-4]  # vector of random weights 


# FUNCTIONS OF ALGORITHM
def Obtain_Exit(v_entry,v_weigth):
    result = np.dot(v_entry, v_weight)
    if(result >= 0):
        y_exit = 1
    if(result < 0):
        y_exit = 0

    return y_exit

def Obtain_Weight(v_entry,v_weigth,error,alpha):
    w1= v_weight[0] + error*alpha*v_entry[0]
    w2= v_weight[1] + error*alpha*v_entry[1]
    w3= v_weight[2] + error*alpha*v_entry[2]
    new_vector=[w1,w2,w3]
    
    return new_vector

# PERCEPTRON 
print("\nLearning....")
while(to_learn == False):
    errors=[]
    for line in training:
        for x in line:
            x=x.split(";")
            v_entry = [1,float(x[0]),float(x[1])]
            y_expected = int(x[2])
            y_obtained = Obtain_Exit(v_entry,v_weight) # get the value between entry and weight
            error=y_expected-y_obtained                # get the error between value expected and value obtained

            if( error != 0 ):
                v_weight= Obtain_Weight(v_entry,v_weight,error,alpha) #change weights to keep learning
                errors.append(1)                                      #append 1 if the error is different of 0
            else:
                errors.append(error)
            plot_errors.append(error)   # save the errors to plot
    
    if 1 not in errors: # if the number 1 doesnt exist in list errors then the perceptron has to break the learning
        to_learn = True 
        print("Done !!")

# MENU 
while(True):
    op2=input("\nSOlVE IRIS VAL   (1)\nPLOT ERRORS      (2)\nEXIT             (3)\n-->")
    if(op2!="1" and op2!="2" and op2!="3"):
            print("ERROR!")
    else:
        if(op2=="1"):
             # READ CSV OF TRAINING
            with open('iris_val.csv', 'r') as f:
                reader = csv.reader(f)
                test = list(reader)
            test.pop(0)
            for line in test:
                for x in line:
                    a=x.split(";")
                    v_entry = [1,float(a[0]),float(a[1])]
                    y_expected = int(a[2])
                    y_obtained = Obtain_Exit(v_entry,v_weight)
                    if(y_obtained==0):
                        print("{",x,"}  Perceptron says: CLASS ",y_obtained )
                    else:
                        print("{",x,"}  Perceptron says: CLASS ",y_obtained )
        if(op2=="2"):
            plt.plot(plot_errors)
            plt.ylim(-1,1)
            plt.show()
        if(op2=="3"):
            break