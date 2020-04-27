import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool

class Net(torch.nn.Module):
    def __init__(self,n_features):
        super(Net, self).__init__()
        self.conv1 = GCNConv(n_features, 128, cached=False) # if you defined cache=True, the shape of batch must be same!
        self.bn1 = BatchNorm1d(128)
        self.conv2 = GCNConv(128, 64, cached=False)
        self.bn2 = BatchNorm1d(64)
        self.fc1 = Linear(64, 64)
        self.bn3 = BatchNorm1d(64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 2)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
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

class Model():
    def __init__(self,num_features,device):
        self.num_features = num_features
        self.device = device
        self.net = Net(num_features)
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