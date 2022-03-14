"""GraphPhage_ice 
Usage:
    GraphPhage_ice.py <fasta_file> <gff_dic> [--out=<fn>] [--thread=<tn>] [--batch=<bn>]
    GraphPhage_ice.py (-h | --help)
    GraphPhage_ice.py --version
Options:
    -o --out=<fn>   The output file name [default: GraphPhage_ice_output.txt].
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


def getHprot(f):
    sizes=[]
    genes=[]
    hprots=[]
    p=''
    G=0
    H=0
    with open(f) as inFile:
        for line in inFile:
            if line.strip()=='##FASTA': # Reached end of annotation
                break
            if line.strip()=="":
                break
            else:
                if line.strip()=='##gff-version 3' or '##sequence-region' in line or "##species https" in line:
                    continue
                else:
                    if line.split()[0]!=p:
                        genes.append(G)
                        hprots.append(H)
                        p=line.split()[0]
                        
                        G=0
                        H=0
                        
                    G+=1
                    # Detect hypothetical proteins
                    for t in line.strip("\n").split('\t')[8].split(';'):
                        if t.split('=')[0]=='product':
                            prod=t.split('=')[1]
                            if prod=='hypothetical protein':
                                H+=1
    genes.append(G)
    hprots.append(H)
    
    genes=genes[1:]
    hprots=hprots[1:]
    
    
    return np.array(hprots)/np.array(genes)


def getGeneDensity(f):
    sizes=[]
    genes=[]
    p=''
    G=0
    with open(f) as inFile:
        for line in inFile:
            if line.strip()=='##FASTA': # Reached end of annotation
                break
            if line.strip()=='': # Reached end of annotation
                break
            else:
                if line.strip()=='##gff-version 3':
                    continue
                if "##species https" in line:
                    continue
                if '##sequence-region' in line: # Read sizes of contigs
                    sizes.append(int(line.split()[-1]))
                else:
                    if line.split()[0]!=p:
                        genes.append(G)
                        p=line.split()[0]
                        G=0
                    G+=1
    genes.append(G)
    genes=genes[1:]
    
    return (np.array(genes)/(np.array(sizes)/1000.0))



def generate_features(fasta_file, gff_dic):
    protein_feature = []
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(seq_record.seq)
        acc = seq_record.id
        hprot = getHprot(gff_dic + "/" + acc + ".gff")[0]
        genedensity = getGeneDensity(gff_dic + "/" + acc + ".gff")[0]
        protein_feature.append([hprot, genedensity])
    protein_feature = np.array(protein_feature) 
    
    return np.array(protein_feature)


class GNN_Model(nn.Module):
    def __init__(self):
        super(GNN_Model, self).__init__()
        self.gcn_dim = 128
        self.cnn_dim = 64
        self.fc_dim = 100
        self.num_layers = 3
        self.dropout = 0.2
        
        self.pnode_d = nn.Linear(4096 * 2, 4096 * 3)
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
        self.d2 = nn.Linear(100 + 2, 200)
        self.d3 = nn.Linear(200, 2)


    def forward(self, data):
        x_f = data.x_src
        x_p = data.x_dst
        protein_info = torch.reshape(data.other_feature, (-1, 2))
        edge_index_forward = data.edge_index[:,::2]
        edge_index_backward = data.edge_index[[1, 0], :][:,1::2]
        # reserve primary nodes dim
        x_p = torch.reshape(x_p, (-1, 4096 * 2))
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
        x = self.d2(torch.cat([x, protein_info], 1))
        x = F.relu(x)
        x = self.d3(x)
        out = F.softmax(x, dim=1)

        return out




def main(arguments):
    fasta_file = arguments.get("<fasta_file>")
    gff_dic = arguments.get("<gff_dic>")
    seq_type = arguments.get("<seq_type>")
    batch_size = int(arguments.get("--batch"))
    output_file = arguments.get("--out")
    thread = int(arguments.get("--thread"))
    
    other_feature = generate_features(fasta_file, gff_dic)
    
    f = open(output_file, "w")
    seq_dic = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(seq_record.seq)
        acc = seq_record.id
        seq_dic[acc] = seq
    seq_name = list(seq_dic.keys())

    data = Biodata.Biodata(dna_seq=seq_dic, other_feature=other_feature, K=3, d=2)
    testset = data.encode(thread=thread)
    print("Make predictions...")
    model = torch.load("model/phage_ice_model.pt", map_location=device)
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
                if each == 0:
                    f.write(seq_name[count] + "\t" + "ice" + "\n")
                elif each == 1:
                    f.write(seq_name[count] + "\t" + "phage" + "\n")
                count += 1
    f.close()
    print("Finish. The result can be find at %s. Thank you for using GraphPhage."%output_file)

if __name__=="__main__":
    arguments = docopt(__doc__, version="GraphPhage_ice 0.1.1")
    main(arguments)
