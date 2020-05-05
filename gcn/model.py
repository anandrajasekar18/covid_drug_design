import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.nn import global_add_pool, global_mean_pool

torch.manual_seed(0)

class ConvNet(torch.nn.Module):
    def __init__(self,n_features):
        super(ConvNet, self).__init__()
        self.conv1 = GCNConv(n_features, 128, cached=False) # if you defined cache=True, the shape of batch must be same!
        self.bn1 = BatchNorm1d(128)
        self.conv2 = GCNConv(128, 64, cached=False)
        self.bn2 = BatchNorm1d(64)
        self.fc1 = Linear(64, 64)
        self.bn3 = BatchNorm1d(64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 2)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        # x = F.softmax(x, dim=1)
        return x 

class EdgeNet(torch.nn.Module):
    '''
    For edge features
    '''
    def __init__(self,num_edge_features,in_channels,out_channels,hidden_dims=[64,64]):
        super(EdgeNet,self).__init__()
        self.lin1 = Linear(num_edge_features,hidden_dims[0])
        self.lin2 = Linear(hidden_dims[0],hidden_dims[1])
        self.lin3 = Linear(hidden_dims[1],in_channels*out_channels)
        
    def forward(self,x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x
    
class EdgeConvNet(torch.nn.Module):
    def __init__(self,n_features):
        super(EdgeConvNet, self).__init__()
        
        self.edgenet1 = EdgeNet(6,n_features,128)
        self.conv1 = NNConv(n_features, 128, nn=self.edgenet1) # if you defined cache=True, the shape of batch must be same!
        self.bn1 = BatchNorm1d(128)
        
        self.edgenet2 = EdgeNet(6,128,64)
        self.conv2 = NNConv(128, 64, nn=self.edgenet2)
        self.bn2 = BatchNorm1d(64)
        
        self.fc1 = Linear(64, 64)
        self.bn3 = BatchNorm1d(64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 2)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # print(x.shape,edge_attr.shape)
        x = F.relu(self.conv1(x, edge_index,edge_attr))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index,edge_attr))
        x = self.bn2(x)
        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        # x = F.softmax(x, dim=1)
        return x 


class Model():
    def __init__(self,num_features,device,net_type):
        self.num_features = num_features
        self.device = device
        if net_type == 'Conv':
            self.net = ConvNet(num_features)
        if net_type == 'Edge':
            self.net = EdgeConvNet(num_features)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)

    def train(self,loader):
        self.net.train()
        loss_all = 0
        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, data.y) # negative log-likelihood loss
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            self.optimizer.step()
        return loss_all / len(loader.dataset)
    
    def test(self,loader):
        self.net.eval()
        correct = 0
        for data in loader:
            data = data.to(self.device)
            output = self.net(data)
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)