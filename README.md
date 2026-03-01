# scEvolver

scEvolver is a continual learning framework designed for single-cell annotation. It is built on a pretrained foundation model and utilizes parameter-efficient fine-tuning (PEFT) to support continual learning.

## Data

The datasets used in this study were sourced from publicly available repositories. Detailed information about the datasets is provided below:

- **PANCREAS dataset**:  
  Includes data from the following studies:  
  - Baron et al. (GSE84133)  
  - Muraro et al. (GSE85241)  
  - Xin et al. (GSE81608)  
  - Segerstolpe et al. (E-MTAB-5061)  
  - Lawlor et al. (GSE86473)  
  The processed human pancreas dataset is available at:  
  [https://github.com/bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)

- **MYELOID dataset**:  
  Publicly available under accession number GSE154763 (Cheng et al., 2021).

- **BMMC dataset**:  
  Obtained from GEO under accession number GSE194122. This dataset includes two sequencing modalities (Luecken et al., 2021).

- **Small Intestinal datasets**:  
  Experimental and control group data are available at:  
  [https://gutcellatlas.org/pangi.html](https://gutcellatlas.org/pangi.html) (Oliver et al., 2024).  
  The datasets were preprocessed using an in-house pipeline, as described in the supplementary materials.

- **Preprocessed Data**:  
  The preprocessed data used in our paper can be downloaded from the following link:  
  [Google Drive Link](https://drive.google.com/drive/folders/1RSrMhZfUkv3TYCoHJfnF_8_Z24UDx-gX?usp=sharing)

  Due to file size limitations, example data can be found here:  
  `../data/PANCREAS/pancreas_batch2.h5ad`

## Model

### Baseline Foundation Model

The baseline foundation model used in scEvolver is based on the Transformer architecture. The pretrained weights can be found at:  
`"../save/scEvolver_human/best_model.pt"`  
You can download the pretrained weights from the original paper:  
[Download Link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y)

### scEvolver PEFT-based Transformer Model

- **PEFT-based Encoder**:  
  The encoder used in scEvolver's PEFT-based model can be found at:  
  `"../tutorials/model/loramoeencoder.py"`

- **Single Modality Transformer Model**:  
  The single modality model is located at:  
  `"../tutorials/model/loramoemodel.py"`

- **Multi-modality Transformer Model**:  
  The multi-modality model can be found at:  
  `"../tutorials/model/loramoemultiomics.py"`

### Prototype-based Continual Learning

The prototype-based continual learning approach is demonstrated in the following script:  
`"../tutorials/platforms_tissues_prototype.py"`

This script includes reference datasets for continual learning, query mapping, outlier detection, and prototype-correlated gene significance analysis.  
For analysis of the prototype-correlated gene significance, refer to:  
`"../tutorials/prototype_analysis.py"`

- **Training Code**: The training function is located in `main()`.  
- **Prediction Code**: The prediction function can be found in `predict()`.

### Trained Weights on Each Task

Trained weights for each task will be released online. Please stay tuned for updates.

---

## Citation

If you use scEvolver in your research, please cite the corresponding paper:

```bibtex
@article{your_paper_here,
  title = {scEvolver: A Continual Learning Framework for Single-Cell Annotation},
  author = {Author 1, Author 2, Author 3},
  journal = {Journal Name},
  year = {Year},
  volume = {Volume},
  pages = {Pages},
}
