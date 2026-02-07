import os
import json
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader
from tqdm import tqdm

from model.fusion_model import HEAL
from model.fusion_model import unified_loss


torch.manual_seed(12345)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = torch.cuda.is_available()

        timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        self.run_dir = os.path.join(args.runs_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        print(f'Run artifacts will be saved in: {self.run_dir}')

        with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    def _load_data(self):
        print('--- Loading Graph Data ---')
        graph_path = os.path.join(self.args.data_root, self.args.graph_filename)
        print(f'Loading graph from: {graph_path}')

        graph_data = torch.load(graph_path, weights_only=False)
        if isinstance(graph_data, dict):
            self.full_graph = HeteroData(graph_data)
        else:
            self.full_graph = graph_data

        for node_type in self.full_graph.node_types:
            node_store = self.full_graph[node_type]
            if (not hasattr(node_store, 'num_nodes')) or (node_store.num_nodes is None):
                if hasattr(node_store, 'x') and node_store.x is not None:
                    node_store.num_nodes = node_store.x.shape[0]

        print('Adding reverse edges...')
        new_edges = {}
        for edge_type in self.full_graph.edge_types:
            src, rel, dst = edge_type
            edge_index = self.full_graph[edge_type].edge_index.contiguous()
            self.full_graph[edge_type].edge_index = edge_index
            rev_rel = f'rev_{rel}'
            rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0).contiguous()
            new_edges[(dst, rev_rel, src)] = rev_edge_index

        for (src, rel, dst), edge_index in new_edges.items():
            self.full_graph[src, rel, dst].edge_index = edge_index

        print(f'Graph now has {len(self.full_graph.edge_types)} edge types (including reverse).')

        import pandas as pd

        if self.args.train_path and self.args.test_path:
            print(f'Loading custom split: {self.args.train_path} | {self.args.test_path}')
            train_df = pd.read_csv(self.args.train_path)
            test_df = pd.read_csv(self.args.test_path)
        else:
            split_dir = os.path.join(self.args.data_root, 'dataset_split')
            print(f'Loading default split from: {split_dir}')
            train_df = pd.read_csv(os.path.join(split_dir, 'disease_disease_train_hybrid.csv'))
            test_df = pd.read_csv(os.path.join(split_dir, 'disease_disease_test_hybrid.csv'))

        train_pairs = train_df[['src', 'dst']].values
        train_labels = train_df['label'].values.astype(np.float32)
        test_pairs = test_df[['src', 'dst']].values
        test_labels = test_df['label'].values.astype(np.float32)

        max_id = max(np.max(train_pairs), np.max(test_pairs))
        dis_nodes = self.full_graph['disease'].num_nodes
        if max_id >= dis_nodes:
            raise ValueError(f'Split IDs out of bounds: max={max_id}, disease_nodes={dis_nodes}')

        num_workers = 4
        self.train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_pairs), torch.from_numpy(train_labels)),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.use_amp,
            persistent_workers=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_pairs), torch.from_numpy(test_labels)),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.use_amp,
            persistent_workers=True,
        )

        print(f'Train samples: {len(train_pairs)}, Test samples: {len(test_pairs)}')

    def _build_model(self):
        print('--- Initializing Model and Optimizer ---')
        self.model = HEAL(
            data=self.full_graph,
            h_dim=self.args.h_dim,
            num_layers=self.args.num_hgt_layers,
            num_heads=self.args.num_attention_heads,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _sample_subgraph(self, batch_pairs):
        input_nodes = batch_pairs.flatten().unique()
        loader = HGTLoader(
            self.full_graph,
            num_samples=self.args.num_neighbors,
            input_nodes=('disease', input_nodes),
            batch_size=len(input_nodes),
            shuffle=False,
            num_workers=0,
        )
        return next(iter(loader))

    def _train_epoch(self, epoch_num):
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch_num + 1}/{self.args.epochs}')
        for batch_pairs, batch_labels in progress_bar:
            subgraph = self._sample_subgraph(batch_pairs).to(self.device)
            batch_pairs = batch_pairs.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                similarity_score, sem_A, task_A, sem_B, task_B = self.model(subgraph, batch_pairs)
                loss, _, _ = unified_loss(
                    similarity_score,
                    batch_labels,
                    sem_A,
                    task_A,
                    sem_B,
                    task_B,
                    beta=self.args.loss_beta,
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        return total_loss / max(1, len(self.train_loader))

    def evaluate(self, is_final_eval=False):
        self.model.eval()
        all_preds_raw = []
        all_labels = []

        with torch.no_grad():
            for batch_pairs, batch_labels in self.test_loader:
                subgraph = self._sample_subgraph(batch_pairs).to(self.device)
                batch_pairs = batch_pairs.to(self.device)

                similarity_score, _, _, _, _ = self.model(subgraph, batch_pairs)
                all_preds_raw.append(similarity_score.cpu().numpy())
                all_labels.append(batch_labels.numpy())

        all_preds_raw = np.concatenate(all_preds_raw).squeeze()
        all_labels = np.concatenate(all_labels)
        all_probs = 1 / (1 + np.exp(-all_preds_raw))
        all_preds_binary = (all_probs > 0.5).astype(int)

        if len(np.unique(all_labels)) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, all_probs, all_labels

        auc = roc_auc_score(all_labels, all_probs)
        aupr = average_precision_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds_binary)
        mcc = matthews_corrcoef(all_labels, all_preds_binary)
        acc = accuracy_score(all_labels, all_preds_binary)
        prec = precision_score(all_labels, all_preds_binary)
        sens = recall_score(all_labels, all_preds_binary)

        if is_final_eval:
            with open(os.path.join(self.run_dir, 'evaluation_results.txt'), 'w') as f:
                f.write(f'AUC: {auc:.4f}\n')
                f.write(f'AUPR: {aupr:.4f}\n')
                f.write(f'F1-Score: {f1:.4f}\n')
                f.write(f'MCC: {mcc:.4f}\n')
                f.write(f'Accuracy: {acc:.4f}\n')
                f.write(f'Precision: {prec:.4f}\n')
                f.write(f'Sensitivity: {sens:.4f}\n')

            np.savetxt(os.path.join(self.run_dir, 'prediction_scores.txt'), all_probs, fmt='%.6f')
            np.savetxt(os.path.join(self.run_dir, 'test_labels.txt'), all_labels, fmt='%d')

        return auc, aupr, f1, mcc, acc, prec, sens, all_probs, all_labels

    def run(self):
        self._load_data()
        self._build_model()
        print(f'Using device: {self.device}')

        best_auc = 0.0
        log_file = os.path.join(self.run_dir, 'training_log.csv')
        with open(log_file, 'w') as f:
            f.write('Epoch,Loss,AUC,AUPR,F1,MCC,Accuracy,Precision,Sensitivity\n')

        for epoch in range(self.args.epochs):
            avg_loss = self._train_epoch(epoch)
            auc, aupr, f1, mcc, acc, prec, sens, _, _ = self.evaluate(is_final_eval=False)

            print(f'Epoch {epoch + 1}: Loss={avg_loss:.4f}, AUC={auc:.4f}, AUPR={aupr:.4f}, F1={f1:.4f}')
            with open(log_file, 'a') as f:
                f.write(
                    f'{epoch + 1},{avg_loss:.6f},{auc:.6f},{aupr:.6f},{f1:.6f},{mcc:.6f},{acc:.6f},{prec:.6f},{sens:.6f}\n'
                )

            if auc > best_auc:
                best_auc = auc
                torch.save(self.model.state_dict(), os.path.join(self.run_dir, 'best_model.pt'))

        best_model_path = os.path.join(self.run_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        self.evaluate(is_final_eval=True)
