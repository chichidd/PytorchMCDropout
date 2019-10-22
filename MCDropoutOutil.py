import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF

#Write a string in the file created in path
def createFile(s,file_name):
    file=open(file_name,"w")
    file.write(str(s)+"\n")
    file.close()

#Write a string in the file whose path is file_name.
def addInfo(s,file_name):
    file=open(file_name,"a")
    file.write(str(s)+"\n")
    file.close()

#save the model to the path (including saved file name).
def saveModel(model,path):
    torch.save(model.state_dict(), path)

#Load an existing model from path
def loadModel(model,path):
    model.load_state_dict(torch.load(path))
    model.eval()

#Normalize the data if necessary
def dataNormalizer(data):
    data_normalized=(data-data.mean())/(1 if data.std()==0 else data.std())
    return data.mean(), 1 if data.std()==0 else data.std() ,data_normalized

#get learning rate for an opitimzer
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#Generate "context" and "target" data for training 
#Note that context points may be seen as training data
def generateData(model,mode,start,end,num_pt=50,length_scale=1.0,N=0,plot=False,file=None):
    np.random.seed(1)
    x=np.random.uniform(start,end,num_pt)[:,None]
    x.sort(axis=0)
    k=RBF(length_scale=length_scale)(x,x)
    y=np.random.multivariate_normal(np.zeros(num_pt),k)[:,None]

    if N==0:
            N=np.random.randint(1, num_pt)

    if mode=='interpol':
        #loc=np.random.binomial(1,float(N/num_pt),num_pt)
        #loc=np.array([bool(i) for i in loc])
        #loc_test=~loc
        pos=np.arange(0,num_pt,int(num_pt/20))
        loc=np.array([])
        loc_test=np.array([])
        pos=np.append(pos,num_pt)
        for i in range(len(pos)):
            if i+1!=len(pos):
                if i%2==1:
                    loc=np.append(loc,np.arange(pos[i],pos[i+1]))
                else:
                    loc_test=np.append(loc_test,np.arange(pos[i],pos[i+1]))
            

        loc=loc.astype(int)
        loc_test=loc_test.astype(int)
    else:
        loc=np.arange(N)
        loc_test=np.arange(N,num_pt)

    if model == "NP":
        x_context=x[loc]
        y_context=y[loc]
        if plot:
            plt.plot(x_context,y_context,'ko',label="context")
            plt.plot(x,y,label="target")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            if file is not None:
                plt.savefig(file)
            plt.show()
        return torch.tensor(x_context).float(),torch.tensor(y_context).float(),torch.tensor(x).float(),torch.tensor(y).float()
    else:
        x_train=x[loc]
        y_train=y[loc]
        x_test=x[loc_test]
        y_test=y[loc_test]
        if plot:
            plt.plot(x_test,y_test,'ro',markersize='1.5',label="test")
            plt.plot(x_train,y_train,'bo',markersize='1.5',label="train")
            
            #plt.plot(x,y)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            if file is not None:
                plt.savefig(file)
            plt.show()
        return torch.tensor(x_train).float(),torch.tensor(y_train).float(),torch.tensor(x_test).float(),torch.tensor(y_test).float()
