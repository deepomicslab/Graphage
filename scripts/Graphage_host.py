"""GraphPhage_host 
Usage:
    GraphPhage_host.py <fasta_file> [--out=<fn>] [--thread=<tn>] [--batch=<bn>]
    GraphPhage_host.py (-h | --help)
    GraphPhage_host.py --version
Options:
    -o --out=<fn>   The output file name [default: GraphPhage_host_output.txt].
    -b --batch=<bn>   The batch size for the prediction process [default: 100].
    -t --thread=<tn>    The number of worker processes to use [default: 10].
    -h --help   Show this screen.
    -v --version    Show version.
"""

from docopt import docopt
import numpy as np
import Biodata
from Bio import SeqIO
from Bio.Seq import Seq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class GNN_Model(nn.Module):
    def __init__(self):
        super(GNN_Model, self).__init__()
        self.gcn_dim = 100
        self.num_layers = 1

        self.pnode_d = nn.Linear(4096 * 3, 4096 * 3)
        torch.nn.init.xavier_uniform_(self.pnode_d.weight)
        torch.nn.init.constant_(self.pnode_d.bias, 0)
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
                self.convs_2.append(pyg_nn.SAGEConv((self.gcn_dim, HIDDEN_DIM), self.gcn_dim))
            else:
                self.convs_2.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))
        self.lns = nn.ModuleList()
        for l in range(self.num_layers-1):
            self.lns.append(nn.LayerNorm(self.gcn_dim))

        self.conv1 = nn.Conv1d(in_channels=self.gcn_dim, out_channels=100, kernel_size=2, bias=False)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=2, bias=False)

        self.d1 = nn.Linear(4094 * 100, 500, bias=False)
        self.d2 = nn.Linear(500, 107, bias=False)


    def forward(self, data):
        x_f = data.x_src
        x_p = data.x_dst
        edge_index_forward = data.edge_index[:,::2]
        edge_index_backward = data.edge_index[[1, 0], :][:,1::2]

        # reserve primary nodes dim
        x_p = torch.reshape(x_p, (-1, 3))

        # change feature nodes dim
        x_f = torch.reshape(x_f, (-1, 64))
        x_f = self.fnode_d(x_f)
        x_f = torch.reshape(x_f, (-1, 3))

        for i in range(self.num_layers):
            x_p = self.convs_1[i]((x_f, x_p), edge_index_forward)
            x_p = F.relu(x_p)
            x_f = self.convs_2[i]((x_p, x_f), edge_index_backward)
            x_f = F.relu(x_f)
            if not i == self.num_layers - 1:
                x_p = self.lns[i](x_p)
                x_f = self.lns[i](x_f)


        x = torch.reshape(x_p, (-1, self.gcn_dim, 4096))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = x.flatten(start_dim = 1)
        x = self.d1(x)
        x = F.relu(x)

        x = self.d2(x)
        out = F.softmax(x, dim=1)

        return out



def main(arguments):
    fasta_file = arguments.get("<fasta_file>")
    seq_type = arguments.get("<seq_type>")
    batch_size = int(arguments.get("--batch"))
    output_file = arguments.get("--out")
    thread = int(arguments.get("--thread"))
    f = open("model/species_label.txt")
    species_label_dic = {}
    for line in f:
        line = line.strip("\n").split("\t")
        species_label_dic[int(line[0])] = line[1]
    f.close()

    f = open(output_file, "w")
    seq_dic = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(seq_record.seq)
        acc = seq_record.id
        seq_dic[acc] = seq
    seq_name = list(seq_dic.keys())

    data = Biodata.Biodata(dna_seq=seq_dic, K=3, d=3)
    testset = data.encode(thread=thread)
    print("Make predictions...")
    model = torch.load("model/phage_host_model.pt", map_location=device)
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
                f.write(seq_name[count] + "\t" + species_label_dic[each] + "\n")
                count += 1
    f.close()
    print("Finish. The result can be find at %s. Thank you for using GraphPhage."%output_file)

if __name__=="__main__":
    arguments = docopt(__doc__, version="GraphPhage_host 0.1.1")
    main(arguments)
