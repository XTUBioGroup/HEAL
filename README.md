# HEAL (PublicVersion)
A Robust Disease Similarity Prediction Method via Semantic-Topological Contrastive Learning.

## Environment
- Python: `3.10+` (recommended `3.10`)
- PyTorch: `2.1+`
- PyTorch Geometric: `2.4+`
- Core packages: `numpy`, `pandas`, `scikit-learn`, `tqdm`

## Install
```bash
pip install numpy pandas scikit-learn tqdm
pip install torch
pip install torch-geometric
```

If you use GPU, install the PyTorch build matching your CUDA version first, then install `torch-geometric`.

## Project Structure
- Entry: `Code/main.py`
- Trainer: `Code/train.py`
- Model: `Code/model/fusion_model.py`

## Run Training
Run from `PublicVersion/Code`:

```bash
python main.py \
  --data_root ../Dataset/Final_Datas \
  --runs_dir ../Experiment/runs/publicversion \
  --graph_filename hetero_graph_v8.0.pt \
  --device 0 \
  --epochs 100 \
  --batch_size 8192
```

Optional custom split:
```bash
python main.py \
  --data_root ../Dataset/Final_Datas \
  --train_path ../Dataset/Final_Datas/dataset_split/disease_disease_train_hybrid.csv \
  --test_path ../Dataset/Final_Datas/dataset_split/disease_disease_test_hybrid.csv
```
# Cite
```
If you use HEAL in your research, please cite:
Jianyi Hu, Xinqiang Wen, Zishan Zhou, Chengqian Lu, Ju Xiang, Ting-Yin Chen,Yongmei Hu*,Xiangmao Meng*.
HEAL: Robust Disease Similarity Prediction via Semantic-TopologicalContrastive Learning[J].XXX, 2026.
```
