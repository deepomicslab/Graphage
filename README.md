# GraphPhage
GraphPhage, a phage analysis tool that incorporates phage and ICE discrimination, phage integration site prediction, phage lifestyle prediction, and phage host prediction. GraphPhage utilizes a Gapped Pattern Graph Convolutional Network (GP-GCN) framework for phage representation learning. The GP-GCN framework is available for various sequence analysis at <https://github.com/deepomicslab/GCNFrame>.

![image](https://github.com/deepomicslab/GraphPhage/blob/main/GraphPhage.png)

## Prerequisite
The scripts are written in Python3. Following Python packages should be installed:
+ cython
+ docopt
+ numpy
+ Biopython
+ pytorch 1.9.0
+ pytorch\_geometric 1.7.0

For phage lifestyle prediction task, please also install Glimmer and Hmmer.
+ Glimmer: conda install -c bioconda glimmer
+ Hmmer: conda install -c bioconda hmmer


## Install
```shell
git clone https://github.com/deepomicslab/GraphPhage.git
cd GraphPhage/scripts
python setup.py build_ext --inplace
```
Please download the model dictionary from <https://drive.google.com/drive/folders/14dXP6VW7d8zNNlHp0Qf_ZbYXvj9qS_Iu?usp=sharing> and put it under the scripts dictionary. 

## Data preparation
For phage integration site prediction, phage lifestyle prediction, and phage host prediction, only fasta file is needed (you can use fasta file with multiple sequences). For phage and ICE discrimination, please provide fasta file and gff file. Please check our example data for details.

## Usage
### Phage and ICE discrimination
```shell
cd GraphPhage/scripts/
python GraphPhage_ice.py phage_genomes.fasta phage_gff_dic --out output.txt --batch 100 --thread 10
```
+ phage\_genomes.fasta contains phage genome sequences in fasta format. GraphPhage supports both single genome and multiple genomes in one file.
+ phage\_gff\_dic contains phage annotations in gff format. Each phage should have one gff file under the dictionary, with the same name as the identity in the fasta file (after >).
+ The input of --out is the filename of the output file (default: GraphPhage\_ice\_output.txt).
+ The input of --thread is the number of worker processes to use for genome encoding (default:10).
+ The input of --batch is the batch size for the prediction process (default:100).

For example:
```shell
python GraphPhage_ice.py ../example_data/example.fasta ../example_data/gff_data
```

There are two columns in the output, with the first showing the sequence id and the second showing the predicted type.

For more information, please use the command:
```shell
python GraphPhage_ice.py -h
```

### Phage integration site prediction
```shell
cd GraphPhage/scripts/
python GraphPhage_int.py phage_genomes.fasta seq_type --out output.txt --batch 100 --thre
ad 10
```
+ phage\_genomes.fasta contains phage genome sequences in fasta format. GraphPhage supports both single genome and multiple genomes in one file.
+ seq\_type should be phage or bacterial, indicating whether your sequences are phage sequences or bactrial sequences.
+ The input of --out is the filename of the output file (default: GraphPhage\_int\_output.txt).
+ The input of --thread is the number of worker processes to use for genome encoding (default:10).
+ The input of --batch is the batch size for the prediction process (default:100).

For example:
```shell
python GraphPhage_int.py ../example_data/example.fasta phage
```
There are three columns in the output, with the first showing the sequence id, the second showing the location on the sequence, and the third showing the prediction (0 for non-integration-site and 1 for integration-site).

For more information, please use the command:
```shell
python GraphPhage_int.py -h
```

### Phage lifestyle prediction
```shell
cd GraphPhage/scripts/
python GraphPhage_lifestyle.py phage_genomes.fasta --out output.txt --batch 100 --thread 10
```
+ phage\_genomes.fasta contains phage genome sequences in fasta format. GraphPhage supports both single genome and multiple genomes in one file.
+ The input of --out is the filename of the output file (default: GraphPhage\_lifestyle\_output.txt).
+ The input of --thread is the number of worker processes to use for genome encoding (default:10).
+ The input of --batch is the batch size for the prediction process (default:100).

For example:
```shell
python GraphPhage_lifestyle.py ../example_data/example.fasta
```

There are two columns in the output, with the first showing the sequence id and the second showing the predicted lifestyle.

For more information, please use the command:
```shell
python GraphPhage_lifestyle.py -h
```
### Phage host prediction
```shell
cd GraphPhage/scripts/
python GraphPhage_host.py phage_genomes.fasta --out output.txt --batch 100 --thread 10
```
+ phage\_genomes.fasta contains phage genome sequences in fasta format. GraphPhage supports both single genome and multiple genomes in one file.
+ The input of --out is the filename of the output file (default: GraphPhage\_host\_output.txt).
+ The input of --thread is the number of worker processes to use for genome encoding (default:10).
+ The input of --batch is the batch size for the prediction process (default:100).

For example:
```shell
python GraphPhage_host.py ../example_data/example.fasta 
```

There are two columns in the output, with the first showing the sequence id and the second showing the predicted host.

For more information, please use the command:
```shell
python GraphPhage_host.py -h
```

## Maintainer
WANG Ruohan ruohawang2-c@my.cityu.edu.hk

