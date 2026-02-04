# scEvolver 
scEvolver is a continual learning framework for single-cell annotation. It is built on pretrained foundation model and applies parameter-efficient fine-tuning (PEFT) to support continual learning. 

# Data
The original datasets were obtained from the Gene Expression Omnibus (GEO) database. The PANCREAS dataset
includes data from Baron (GSE84133), Muraro (GSE85241), Xin (GSE81608), Segerstolpe (E-MTAB-5061), and
Lawlor (GSE86473). The processed human pancreas dataset was retrieved from https://github.com/bowang-lab/
scGPT. The MYELOID dataset is publicly available under accession number GSE154763 Cheng et al. [2021]. The
BMMC dataset was obtained from GEO under accession number GSE194122 and includes two sequencing modalities
Luecken et al. [2021]. The small intestinal datasets for the experimental and control groups were obtained from
https://gutcellatlas.org/pangi.html Oliver et al. [2024] and were preprocessed using an in-house pipeline,
as described in the supplementary materials. 
The preprocessed data in our paper can be downloaded from: we will released here. Due to file size limitations, the example data can be found in: 
../data/PANCREAS/pancreas_batch2.h5ad

# Model
## Baseline foundation model:
from scgpt.model import TransformerModel
Pretrained weights from: "../save/scEvolver_human/best_model.pt", which can be downloaded from original paper:  
https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y

## scEvolver PEFT-based Transformer model
PEFT-based encoder:
"../tutorials/model/loramoeencoder.py"

Single modality Transformer model:
"../tutorials/model/loramoemodel.py"

Multi-modality Transformer model:
"../tutorials/model/loramoemultiomics.py"

## Prototype-based Continual Learning
The script "../tutorials/platforms_tissues_prototype.py" demonstrates reference datasets continual learning, query mapping, outlier detection, and prototype-correlated gene significance analysis ("../tutorials/prototype_analysis.py").
The training code is in function 'main()', and the prediction code can be found in function 'predict()'. 

## Trained weights on each tasks
The trained weights on each tasks will be released online.
