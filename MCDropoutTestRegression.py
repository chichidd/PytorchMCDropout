import torch
import torch.utils.data
import MCDropoutOutil
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

#calculate RMSE
def test(Trainer,Xtest,ytest):
    #make model in training mode to do dropout
    Trainer.model.eval()
    #normalize xdata
    if Trainer.normalize:
        _,__,Xtest=MCDropoutOutil.dataNormalizer(Xtest)
    #put it to the device and change its form to pass the network
    xdata=Xtest.to(Trainer.device).view(Xtest.shape[0],-1)
    ytest=ytest.to(Trainer.device)
    #make prediction 
    y_pred=Trainer.model(xdata)
    y_pred=y_pred*Trainer.ytrain_std+Trainer.ytrain_mean
    
    #calculate loss for the test
    loss=Trainer.criterion(y_pred,ytest)
    loss/=ytest.shape[0]
    
    #print info: RMSE, R2 and loss
    y_pred=y_pred.detach().numpy()
    MCDropoutOutil.addInfo("Prediction:",Trainer.loss_file)
    MCDropoutOutil.addInfo("RMSE:{}, R2 score:{}, loss:{}".format(mean_squared_error(ytest,y_pred),r2_score(ytest,y_pred),loss),Trainer.loss_file)



#calculate RMSE with uncertainty
def test_uncertainty(Trainer,Xtest,ytest,Xtrain,ytrain,plotpath=None):
    #make model in training mode to do dropout
    Trainer.model.train()
    #normalize x
    if Trainer.normalize:
        _,__,Xtest=MCDropoutOutil.dataNormalizer(Xtest)
    #bug
    
    Xtest=Xtest.to(Trainer.device).view(Xtest.shape[0],-1)
    ytest=ytest.to(Trainer.device)
    
    #concatenate train and test
    X=torch.cat([Xtrain,Xtest],dim=0).view(-1)
    y=torch.cat([ytrain,ytest],dim=0).view(-1)
    sortindice=torch.sort(X).indices
    y=y[sortindice]
    X=X[sortindice]
    
    #average over output direct instead of class predicted
    outputlist=[]
    for i in range(Trainer.T):
        outputlist.append(torch.unsqueeze(Trainer.model(X.view(-1,1)),0))
    output_cat=torch.cat(outputlist,dim=0)

    y_pred=output_cat.mean(dim=0).squeeze()
    #multiply std and add mean to compare with ytest
    y_pred=y_pred*Trainer.ytrain_std+Trainer.ytrain_mean

    #calculate std
    std_pred=output_cat.std(dim=0)

    #calculate loss for the test
    loss=Trainer.criterion(y_pred,y)

    #calculate log likelihood approximation
    output_cat_for_norm=(output_cat.view(Trainer.T,-1,1)-y.view(-1,1).expand(Trainer.T,-1,1)).view(Trainer.T,-1)

    ll=torch.logsumexp(-0.5 * Trainer.tau * torch.norm(output_cat_for_norm,dim=1),dim=0) - np.log(Trainer.T) - 0.5*np.log(2*np.pi) + 0.5*np.log(Trainer.tau)
    loss/=ytest.shape[0]
    #print info: RMSE, R2, log likelihood approximation (eq. 8) and loss

    y_pred=y_pred.view(-1).detach().numpy()
    

    MCDropoutOutil.addInfo("Prediction with uncertainty",Trainer.loss_file)
    MCDropoutOutil.addInfo("RMSE:{}, R2 score:{}, ll:{}, loss:{}".format(mean_squared_error(y,y_pred),r2_score(y,y_pred),ll.detach().item(),loss),Trainer.loss_file)

    if plotpath is not None:
        Xtest=Xtest.view(-1).detach().numpy()
        ytest=ytest.view(-1).detach().numpy()
        ytrain=ytrain.detach().numpy()
        Xtrain=Xtrain.detach().numpy()

        X=X.view(-1).detach().numpy()
        y=y.view(-1).detach().numpy()
        std_pred=(std_pred*Trainer.ytrain_std).view(-1).detach().numpy()
        
        y_up=y_pred+std_pred
        y_down=y_pred-std_pred

        
        
        plt.plot(Xtrain,ytrain,'bo',markersize='1.5',label="train")
        plt.plot(Xtest,ytest,'ro',markersize='1.5',label="test")
        plt.plot(X,y_pred,color="g",label="prediction")
        
        plt.fill_between(X,y_down,y_up,color='g',alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(plotpath)
        return X,y_pred
