import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

class DropoutNetwork(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_layers,proba_drop=0.5,normDist=False):
        #assign module after called Module.__init__() 
        #if normDist==True, original distribution of weights is gaussien
        super(DropoutNetwork, self).__init__()
        
        #basic parameters
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.proba_drop=proba_drop
        
        
        
        #construction of hidden layers
        last_layer_num=input_dim
        self.hidden_layer_list=[]
        for i,l in enumerate(hidden_layers):
            self.hidden_layer_list.append(('dense'+str(i+1),nn.Linear(last_layer_num,l)))
            if normDist:
                nn.init.normal_(self.hidden_layer_list[i*2][1].weight)
            self.hidden_layer_list.append(('dropout'+str(i+1),nn.Dropout(p=proba_drop)))
            self.hidden_layer_list.append(('relu'+str(i+1),nn.ReLU()))
            
            last_layer_num=l

        self.hidden_layer=nn.Sequential(OrderedDict(self.hidden_layer_list))
        #construction of output layers
        self.layer_out=nn.Linear(last_layer_num,self.output_dim)
        if normDist:
            nn.init.normal_(self.layer_out.weight)
        
        
        
    def forward(self, xdata,drop_inplace=False):
        
        input=xdata.view(-1,self.input_dim)
        
        #Pass layers with dropout
        input=self.hidden_layer(input)
        #Pass layer 3 to get the output
        output=self.layer_out(input)
        
        return output