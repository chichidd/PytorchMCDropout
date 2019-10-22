import torch
import torch.utils.data
import MCDropoutOutil
from sklearn.metrics import confusion_matrix,accuracy_score


#calculate accuracy score and confusion matrix
def test_class(Trainer,Xtest,ytest):
    #make model in training mode to do dropout
    
    Trainer.model.train()
    loss=0
    #load test data
    kwargs = {'num_workers': 1, 'pin_memory': True} if Trainer.cuda else {} #parameter useful for cuda (GPU)
    test_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtest, ytest),batch_size=Trainer.batch_size,shuffle=True,**kwargs)
    #two list y_pred and y_true used for register
    y_pred=[]
    y_true=[]
    for batch_idx, (xdata,ydata) in enumerate(test_loader):
        xdata=xdata.to(Trainer.device).view(xdata.shape[0],-1)
        ydata=ydata.to(Trainer.device)
        
        
        outputlist=[]
        #average over output direct instead of class predicted
        for i in range(Trainer.T):
            outputlist.append(torch.unsqueeze(Trainer.model(xdata),0))
        output_mean=torch.cat(outputlist,dim=0).mean(dim=0)
        #calculate loss for the test
        loss+=Trainer.criterion(output_mean,ydata)
        #predict the class: index of the highest softmax probability
        pred=output_mean.max(dim=1,keepdim=True)[1]
        
        y_pred.append(pred,dim=0)
        y_true.append(ydata,dim=0)
        if (batch_idx+1) % (int(len(test_loader)/10))==0:
            MCDropoutOutil.addInfo("Test finished {:.0f}%".format(100. * (batch_idx+1) / len(test_loader)),Trainer.loss_file)
    loss/=len(test_loader)
    y_pred=torch.cat(y_pred,dim=0)
    y_true=torch.cat(y_true,dim=0)
    #print info: accuracy score and confusion matrix
    MCDropoutOutil.addInfo('\nMC Dropout Test set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n Confusion matrix:'.format(
        loss, 
        accuracy_score(y_true,y_pred)),Trainer.loss_file)
    MCDropoutOutil.addInfo(confusion_matrix(y_true,y_pred),Trainer.loss_file)

