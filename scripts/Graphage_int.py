"""GraphPhage_int 
Usage:
    GraphPhage_int.py <fasta_file> <seq_type> [--out=<fn>] [--thread=<tn>] [--batch=<bn>]
    GraphPhage_int.py (-h | --help)
    GraphPhage_int.py --version
Options:
    -o --out=<fn>   The output file name [default: GraphPhage_int_output.txt].
    -b --batch=<bn>   The batch size for the prediction process [default: 100].
    -t --thread=<tn>    The number of worker processes to use [default: 10].
    -h --help   Show this screen.
    -v --version    Show version.
"""

from docopt import docopt
import numpy as np
import Biodata
from Bio import SeqIO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def generate_subsequence(seq, sw_length):
    seq_dic = {}
    for i in range(0, len(seq)-sw_length+1):
        seq_dic[i+301] = seq[i: i+sw_length]
    
    return seq_dic

class GNN_Model(nn.Module):
    def __init__(self):
        super(GNN_Model, self).__init__()
        self.gcn_dim = 128
        self.cnn_dim = 64
        self.fc_dim = 100
        self.num_layers = 3
        self.dropout = 0.2

        self.pnode_d = nn.Linear(4096 * 3, 4096 * 3)
        self.fnode_d = nn.Linear(64, 64 * 3)

        self.convs_1 = nn.ModuleList()
        for l in range(self.num_layers):
            if l == 0:
                self.convs_1.append(pyg_nn.SAGEConv((3, 3), self.gcn_dim))
            else:
                self.convs_1.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))

        self.convs_2 = nn.ModuleList()
        for l in range(self.num_layers):
            if l == 0:
                self.convs_2.append(pyg_nn.SAGEConv((self.gcn_dim, 3), self.gcn_dim))
            else:
                self.convs_2.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))

        self.lns = nn.ModuleList()
        for l in range(self.num_layers-1):
            self.lns.append(nn.LayerNorm(self.gcn_dim))

        self.conv1 = nn.Conv1d(in_channels=self.gcn_dim, out_channels=64, kernel_size=8)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8)
        self.d1 = nn.Linear(4075 * 64, 100)
        self.d2 = nn.Linear(100, 2)


    def forward(self, data):
        x_f = data.x_src
        x_p = data.x_dst
        edge_index_forward = data.edge_index[:,::2]
        edge_index_backward = data.edge_index[[1, 0], :][:,1::2]

        # reserve primary nodes dim
        x_p = torch.reshape(x_p, (-1, 4096 * 3))
        x_p = self.pnode_d(x_p)
        x_p = torch.reshape(x_p, (-1, 3))

        # change feature nodes dim
        x_f = torch.reshape(x_f, (-1, 64))
        x_f = self.fnode_d(x_f)
        x_f = torch.reshape(x_f, (-1, 3))

        for i in range(self.num_layers):
            x_p = self.convs_1[i]((x_f, x_p), edge_index_forward)
            x_p = F.relu(x_p)
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)
            x_f = self.convs_2[i]((x_p, x_f), edge_index_backward)
            x_f = F.relu(x_f)
            x_f = F.dropout(x_f, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x_p = self.lns[i](x_p)
                x_f = self.lns[i](x_f)

        x = torch.reshape(x_p, (-1, self.gcn_dim, 4096))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.flatten(start_dim = 1)
        x = self.d1(x)
        x = F.relu(x)

        logits = self.d2(x)
        out = F.softmax(logits, dim=1)

        return out



def main(arguments):
    fasta_file = arguments.get("<fasta_file>")
    seq_type = arguments.get("<seq_type>")
    batch_size = int(arguments.get("--batch"))
    output_file = arguments.get("--out")
    thread = int(arguments.get("--thread"))
    f = open(output_file, "w")
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(seq_record.seq)
        acc = seq_record.id
        seq_dic = generate_subsequence(seq, 600)
        seq_name = list(seq_dic.keys())
        data = Biodata.Biodata(dna_seq=seq_dic, K=3, d=3)
        testset = data.encode(thread=thread)
        print("Make predictions...")
        if seq_type == "phage":
            model = torch.load("model/phage_int_model.pt", map_location=device)
        elif seq_type == "bacterial":
            model = torch.load("model/bac_int_model.pt", map_location=device)
        for i in range(1, 301):
            f.write(acc + "\t" + str(i) + "\t" + "0" + "\n")
        loader = DataLoader(testset, batch_size=batch_size, shuffle=False, follow_batch=['x_src', 'x_dst'])
        model.eval()
        count = 0
        for data in loader:
            with torch.no_grad():
                data = data.to(device)
                pred = model(data)
                pred = pred.argmax(dim=1)
                pred = pred.cpu().numpy()
                for each in pred:
                    f.write(acc + "\t" + str(seq_name[count]) + "\t" + str(each) + "\n")
                    count += 1
        for i in range(count+301, count+600):
            f.write(acc + "\t" + str(i) + "\t" + "0" + "\n")

    f.close()
    print("Finish. The result can be find at %s. Thank you for using GraphPhage."%output_file)

if __name__=="__main__":
    arguments = docopt(__doc__, version="GraphPhage_int 0.1.1")
    main(arguments)
