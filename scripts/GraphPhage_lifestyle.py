"""GraphPhage_lifestyle 
Usage:
    GraphPhage_lifestyle.py <fasta_file> [--out=<fn>] [--thread=<tn>] [--batch=<bn>]
    GraphPhage_lifestyle.py (-h | --help)
    GraphPhage_lifestyle.py --version
Options:
    -o --out=<fn>   The output file name [default: GraphPhage_lifestyle_output.txt].
    -b --batch=<bn>   The batch size for the prediction process [default: 100].
    -t --thread=<tn>    The number of worker processes to use [default: 10].
    -h --help   Show this screen.
    -v --version    Show version.
"""

import os
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


def generate_features(fasta_file):

    print("Generating protein features...")
    os.system("mkdir tmp")
    os.system("long-orfs -z 11 -n -t 1.15 -l %s tmp/tmp.longorfs"%fasta_file)
    os.system("extract -t %s tmp/tmp.longorfs > tmp/tmp.train"%fasta_file)
    os.system("build-icm -r tmp/tmp.icm < tmp/tmp.train")
    os.system("glimmer3 -z 11 -o 50 -g 110 -t 30 -l %s tmp/tmp.icm tmp/tmp"%fasta_file)
    f_in = open("tmp/tmp.predict")
    f_out = open("tmp/tmp.predict_nor", "w")
    for line in f_in:
        if line.startswith(">"):
            acc = line.strip(">").strip("\n").split(" ")[0]
        else:
            line = line.split()
            f_out.write(line[0] + " ")
            f_out.write(acc + " ")
            f_out.write(line[1] + " ")
            f_out.write(line[2] + " ")
            f_out.write(line[3] + " ")
            f_out.write(line[4] + "\n")
    f_in.close()
    f_out.close()
    os.system("multi-extract -t %s tmp/tmp.predict_nor > tmp/tmp.cds"%fasta_file)
    f_in = open("tmp/tmp.cds")
    f_out = open("tmp/tmp.aa",'w')
    seq  = ''
    for line in f_in:
        if line.startswith('>'):
            if seq != '':
                seq = Seq(seq)
                aa = seq.translate()
                aa = str(aa)
                aa = 'M'+aa[1:]
                f_out.write(info)
                while len(aa)>60:
                    f_out.write(aa[:60]+'\n')
                    aa = aa[60:]
                else:
                    f_out.write(aa + '\n')
            info = line
            seq = ''
        else:
            seq = seq + line.strip('\n')
    f_in.close()
    f_out.close()
    os.system("hmmsearch --tblout tmp/result.txt --noali model/prot_models.hmm tmp/tmp.aa")
    protein_dic = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        protein_dic[seq_record.id] = []

    f_in = open("tmp/result.txt")

    for line in f_in:
        if not line.startswith("#"):
            line = line.strip("\n").split()
            protein_dic[line[-4]].append(line[2])
    f_in.close()

    all_protein_list = np.loadtxt("model/CDD_protein_list.tsv", dtype="str")
    f_out = open("tmp/CDD_protein_feature.txt", "w")
    for each in protein_dic:
        for pro in all_protein_list:
            if pro in protein_dic[each]:
                f_out.write("1" + "\t")
            else:
                f_out.write("0" + "\t")
        f_out.write("\n")
    f_out.close()
    protein_feature = np.loadtxt("tmp/CDD_protein_feature.txt")
    os.system("rm -r tmp")
    
    return protein_feature


class GNN_Model(nn.Module):
    def __init__(self):
        super(GNN_Model, self).__init__()
        self.gcn_dim = 128
        self.cnn_dim = 64
        self.fc_dim = 100
        self.num_layers = 4
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
        self.d2 = nn.Linear(100 + 206, 100 + 206)
        self.d3 = nn.Linear(100 + 206, 2)


    def forward(self, data):
        x_f = data.x_src
        x_p = data.x_dst
        protein_info = torch.reshape(data.other_feature, (-1, 206))
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
        x = self.d2(torch.cat([x, protein_info], 1))
        x = F.relu(x)
        x = self.d3(x)
        out = F.softmax(x, dim=1)

        return out


def main(arguments):
    fasta_file = arguments.get("<fasta_file>")
    seq_type = arguments.get("<seq_type>")
    batch_size = int(arguments.get("--batch"))
    output_file = arguments.get("--out")
    thread = int(arguments.get("--thread"))
    
    other_feature = generate_features(fasta_file)
    
    f = open(output_file, "w")
    seq_dic = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(seq_record.seq)
        acc = seq_record.id
        seq_dic[acc] = seq
    seq_name = list(seq_dic.keys())

    data = Biodata.Biodata(dna_seq=seq_dic, other_feature=other_feature, K=3, d=3)
    testset = data.encode(thread=thread)
    print("Make predictions...")
    model = torch.load("model/phage_lifestyle_model.pt", map_location=device)
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
                    f.write(seq_name[count] + "\t" + "virulent" + "\n")
                elif each == 1:
                    f.write(seq_name[count] + "\t" + "temperate" + "\n")
                count += 1
    f.close()
    print("Finish. The result can be find at %s. Thank you for using GraphPhage."%output_file)

if __name__=="__main__":
    arguments = docopt(__doc__, version="GraphPhage_lifestyle 0.1.1")
    main(arguments)
