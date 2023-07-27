#Graph to csv
import torch
from collections import defaultdict
import torch_geometric
import numpy as np
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import pandas as pd
import random
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from torch_geometric.data import InMemoryDataset, Data
import torch.nn.functional as f
from torch.utils.data import random_split

class DatasetCreation():
    def __init__(self):

        if torch.cuda.is_available():
            self.device = "cuda:3"
            print("Device is: ",self.device)
        else:
            self.device = "cpu"
        self.device = "cpu"

    def GraphtoCSV(self, name, Normalize=None, Weights=None):
        G = nx.read_gpickle(name)
        print("Total Nodes:",G.number_of_nodes())
        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous() #'userID','screen_name',
        group_node_attrs = ['followers_count', 'friends_count', 'listed_count', 'acc_created_at',
                            'favourites_count', 'verified', 'Tweets',
                            'followerPerDay', 'statusPerDay', 'favPerTweet', 'friendsPerDay', 'followFriends',
                            'friendNlisted','prot','foPerTweet','frPerTweet','favPerFollow','favPerFriend','listPerDay','Exists']
        data = defaultdict(list)
        group_edge_attrs=None
        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
            print("The orignal node attributes", node_attrs)
        else:
            node_attrs = {}

        if G.number_of_edges() > 0:
            edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
        else:
            print("No edge attributes")
            edge_attrs = {}

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            if set(feat_dict.keys()) != set(node_attrs):
                raise ValueError('Not all nodes contain the same attributes')
            for key, value in feat_dict.items():
                # print(key,value)
                data[str(key)].append(value)

        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            if set(feat_dict.keys()) != set(edge_attrs):
                raise ValueError('Not all edges contain the same attributes')
            for key, value in feat_dict.items():
                data[str(key)].append(value)
        for key, value in data.items():
            try:
                data[key] = torch.tensor(value)
            except ValueError:
                pass
        data = torch_geometric.data.Data.from_dict(data)
        x_np = np.empty((len(G.nodes), len(node_attrs)), dtype=float)
        i = 0
        for n in G.nodes:
            onenode = []
            for a in node_attrs:
                # print(Gp.nodes[n][a])
                onenode.append(G.nodes[n][a])
            # print(onenode)
            x_np[i] = onenode
            i = i + 1
        data.x = torch.from_numpy(x_np)  # numpy to torch
        temp= data.x.select(1,13)#[7:7]
        temp_np=temp.detach().numpy()
        data.y=torch.from_numpy(temp_np).float()
        data=data.to(self.device)
        data.x=data.x.to(self.device)
        if data.x is None:
            data.num_nodes = G.number_of_nodes()
        if group_node_attrs is all:
            group_node_attrs = list(node_attrs)
        if group_node_attrs is not None:
            xs = [data[key] for key in group_node_attrs]
            xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
            data.x = torch.cat(xs, dim=-1)
        data.num_classes = 2
        if group_edge_attrs is all:
            group_edge_attrs = list(edge_attrs)
        if group_edge_attrs is not None:
            edge_attrs = [data[key] for key in group_edge_attrs]
            edge_attrs = [x.view(-1, 1) if x.dim() <= 1 else x for x in edge_attrs]
            data.edge_attr = torch.cat(edge_attrs, dim=-1).float()
            print("data edge_attr is using", data.edge_attr.get_device())
        if Normalize == True:
            # Data
            print("Data is Normalized")
            transformer = MaxAbsScaler().fit(x_np)
            newdf = transformer.transform(x_np)
            import pandas as pd
            px = pd.DataFrame(newdf)#'userID','screen_name',
            px.columns = ['followers_count', 'friends_count', 'listed_count', 'acc_created_at',
                            'favourites_count', 'verified', 'Tweets',
                            'followerPerDay', 'statusPerDay', 'favPerTweet', 'friendsPerDay', 'followFriends',
                            'friendNlisted','prot','foPerTweet','frPerTweet','favPerFollow','favPerFriend','listPerDay','y']
            px.to_csv("CSVFiles/IASC AllUsers Normalized.csv", index=False)
            data.x = torch.from_numpy(newdf).float()
        else:
            import pandas as pd
            px = pd.DataFrame(x_np)#'userID','screen_name',
            px.columns = ['followers_count', 'friends_count', 'listed_count', 'acc_created_at',
                                'favourites_count', 'verified', 'Tweets',
                                'followerPerDay', 'statusPerDay', 'favPerTweet', 'friendsPerDay', 'followFriends',
                                'friendNlisted','prot','foPerTweet','frPerTweet','favPerFollow','favPerFriend','listPerDay','y']
            px.to_csv("CSVFiles/IASC AllUsers.csv", index=False)
        print("CSVFiles/IASC AllUsers.csv created")



