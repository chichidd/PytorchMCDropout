import torch
from torch import nn, optim
import datetime
import MCDropoutNetwork
import MCDropoutOutil


class MCDropoutTrainer():

    def __init__(self, device, Xtrain, ytrain, optimizer, criterion, batch_size=256, n_epochs=300, model=None, tau=1.0,
        length_scale=1e-2, T=100, normalize=False, print_freq=10, result_path="./", loss_file="./loss",cuda=False):
        self.device = device
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {} #parameter useful for cuda (GPU)
        self.normalize=normalize
        if normalize:
            _,__,self.Xtrain=MCDropoutOutil.dataNormalizer(Xtrain)
            self.ytrain_mean,self.ytrain_std,self.ytrain=MCDropoutOutil.dataNormalizer(ytrain)
            
        else:
            self.Xtrain=Xtrain
            self.ytrain=ytrain
            self.ytrain_mean=0
            self.ytrain_std=1

        self.batch_size=batch_size
        self.n_epochs=n_epochs


        self.optimizer = optimizer
        self.criterion=criterion
        self.print_freq = print_freq

        self.result_path=result_path
        self.loss_file=loss_file
        
        if model is not None:
            self.model = model
        else:
            self.model=MCDropoutNetwork.DropoutNetwork(Xtrain.shape[1],ytrain.shape[1],hidden_layers=[256,256],proba_drop=0.5)

        self.lambda_reg=length_scale**2*(1-self.model.proba_drop)/(2.*self.Xtrain.shape[0]*tau)#We consider a same length scale for each layer's weight
        self.cuda=cuda
        self.T=T
        self.tau=tau

    def train(self,printInfo=False):
        self.model.train()
        MCDropoutOutil.createFile("Begin training...",self.loss_file)
        min_loss=10000
        
        #set training data loader
        
        train_loader=torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.Xtrain, self.ytrain),
                batch_size=self.batch_size,shuffle=True,**self.kwargs)
        #update learning rate : reduce lr when loss has stopped decreasing
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min',factor=0.8,patience=30)
        for epoch in range(self.n_epochs):
            time1=datetime.datetime.now()
            epoch_loss=0
            
            for batch_idx, (X,y) in enumerate(train_loader):
                
                #if we have cuda
                X=X.to(self.device)
                y=y.to(self.device)

                self.optimizer.zero_grad()
                output=self.model(X)
                loss=self.loss_func(output,y)
                epoch_loss+=loss.item()
                loss.backward()
                self.optimizer.step()
                
                if printInfo and batch_idx % self.print_freq==0:
                    MCDropoutOutil.addInfo('Train Epoch: {} [{}/{} ({:.0f}%)] lr: {}\tLoss: {:.6f}\n'
                    .format(epoch+1, batch_idx * len(y),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            MCDropoutOutil.get_lr(self.optimizer), loss.item()),self.loss_file)
            
            #write epoch information in the file
            MCDropoutOutil.addInfo("Finish {} epoch(s). Epoch loss: {}. ".format(epoch+1,epoch_loss/len(train_loader))+" Time of epoch:"+str(datetime.datetime.now()-time1)+"\n",self.loss_file)
            #update learning rate
            lr_scheduler.step(epoch_loss/len(train_loader))
            #save the model if loss is lower
            if epoch_loss/len(train_loader) < min_loss:
                min_loss=epoch_loss/len(train_loader)
                MCDropoutOutil.saveModel(self.model,self.result_path+"model_{}.pt".format(epoch+1))         
            
    def loss_func(self, output,y,setReguInOptimizer=False):
        loss=self.criterion(output,y)

        #Note that if we have a universal length-scale, we can set parameter of regularisation (lambda) in optimizer
        #calculate regularisation terms manually
        if setReguInOptimizer == False:
            for _, param in self.model.named_parameters():
                loss+=self.lambda_reg*param.norm(2)
        return loss

