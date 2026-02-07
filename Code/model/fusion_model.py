import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class SemanticRouter(nn.Module):
    def __init__(self, semantic_dim, h_dim, num_experts):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(semantic_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_experts),
        )

    def forward(self, semantic_features):
        logits = self.router(semantic_features)
        return F.softmax(logits, dim=-1)


class HEAL(nn.Module):
    def __init__(self, data, h_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.h_dim = h_dim
        self.node_types = data.node_types

        self.expert_keys = [nt for nt in self.node_types if nt != 'disease']
        self.num_experts = len(self.expert_keys)

        self.node_embeds = nn.ModuleDict()
        self.feature_encoders = nn.ModuleDict()
        self.semantic_dim = 0

        for node_type in self.node_types:
            feat_dim = 1
            if data[node_type].x is not None:
                feat_dim = data[node_type].x.shape[1]

            if node_type == 'disease' and feat_dim > 1:
                self.semantic_dim = feat_dim

            if feat_dim > 1:
                self.feature_encoders[node_type] = nn.Sequential(
                    nn.Linear(feat_dim, h_dim),
                    nn.ReLU(),
                    nn.LayerNorm(h_dim),
                )
            else:
                self.node_embeds[node_type] = nn.Embedding(data[node_type].num_nodes, h_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(h_dim, h_dim, data.metadata(), num_heads))

        if self.semantic_dim > 0:
            self.router = SemanticRouter(self.semantic_dim, h_dim, self.num_experts)
            self.semantic_proj = nn.Linear(self.semantic_dim, h_dim)
        else:
            self.router_bias = nn.Parameter(torch.randn(1, self.num_experts))

        self.expert_proj = nn.ModuleDict()
        for key in self.expert_keys:
            self.expert_proj[key] = nn.Linear(h_dim, h_dim)

        self.predictor = nn.Sequential(
            nn.Linear(h_dim * 4, h_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h_dim, 1),
        )

        self.contrastive_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )

    def forward(self, subgraph, batch_pairs_local):
        raw_semantic_feats = None
        if 'disease' in subgraph.node_types and subgraph['disease'].x is not None:
            raw_semantic_feats = subgraph['disease'].x

        x_dict = {}
        for nt in subgraph.node_types:
            if nt in self.feature_encoders:
                x_dict[nt] = self.feature_encoders[nt](subgraph[nt].x)
            else:
                x_dict[nt] = self.node_embeds[nt](subgraph[nt].n_id)

        if raw_semantic_feats is not None and hasattr(self, 'semantic_proj') and 'disease' in x_dict:
            x_dict['disease'] = x_dict['disease'] + self.semantic_proj(raw_semantic_feats)

        for conv in self.convs:
            with torch.cuda.amp.autocast(enabled=False):
                x_dict = {k: v.float() for k, v in x_dict.items()}
                x_dict = conv(x_dict, subgraph.edge_index_dict)

        disease_topology_embeds = x_dict['disease']
        disease_expert_embeds, _ = self._compute_semantic_guided_experts(subgraph, x_dict, raw_semantic_feats)

        global_ids = subgraph['disease'].n_id
        map_g2l = {gid.item(): i for i, gid in enumerate(global_ids)}

        valid_pairs_list = []
        for p in batch_pairs_local:
            src, dst = p[0].item(), p[1].item()
            if src in map_g2l and dst in map_g2l:
                valid_pairs_list.append([map_g2l[src], map_g2l[dst]])

        if not valid_pairs_list:
            empty = torch.tensor(0.0, device=disease_topology_embeds.device)
            return empty, None, None, None, None

        local_pairs = torch.tensor(valid_pairs_list, device=disease_topology_embeds.device)

        topo_A = disease_topology_embeds[local_pairs[:, 0]]
        topo_B = disease_topology_embeds[local_pairs[:, 1]]
        expert_A = disease_expert_embeds[local_pairs[:, 0]]
        expert_B = disease_expert_embeds[local_pairs[:, 1]]

        proj_topo_A = self.contrastive_head(topo_A)
        proj_topo_B = self.contrastive_head(topo_B)
        proj_expert_A = self.contrastive_head(expert_A)
        proj_expert_B = self.contrastive_head(expert_B)

        predictor_input = torch.cat([
            topo_A,
            topo_B,
            torch.abs(topo_A - topo_B),
            topo_A * topo_B,
        ], dim=-1)

        similarity_score = self.predictor(predictor_input)
        return similarity_score, proj_topo_A, proj_expert_A, proj_topo_B, proj_expert_B

    def _compute_semantic_guided_experts(self, subgraph, all_embeds, raw_semantic_feats):
        device = all_embeds['disease'].device
        num_diseases = all_embeds['disease'].size(0)

        if raw_semantic_feats is not None and hasattr(self, 'router'):
            alpha = self.router(raw_semantic_feats)
        elif hasattr(self, 'router_bias'):
            alpha = F.softmax(self.router_bias, dim=-1).expand(num_diseases, -1)
        else:
            alpha = torch.ones(num_diseases, self.num_experts, device=device) / self.num_experts

        total_expert_context = torch.zeros(num_diseases, self.h_dim, device=device, dtype=all_embeds['disease'].dtype)

        for i, expert_key in enumerate(self.expert_keys):
            if expert_key not in all_embeds:
                continue

            target_edge_type = None
            reverse = False
            for edge_t in subgraph.edge_types:
                if edge_t[0] == expert_key and edge_t[2] == 'disease':
                    target_edge_type = edge_t
                    reverse = False
                    break
                if edge_t[0] == 'disease' and edge_t[2] == expert_key:
                    target_edge_type = edge_t
                    reverse = True
                    break

            if target_edge_type is None:
                continue

            edge_index = subgraph[target_edge_type].edge_index
            if reverse:
                edge_index = torch.stack([edge_index[1], edge_index[0]])

            src, dst = edge_index[0], edge_index[1]
            expert_h = self.expert_proj[expert_key](all_embeds[expert_key])
            msg = expert_h[src]

            expert_summary = torch.zeros(num_diseases, self.h_dim, device=device, dtype=msg.dtype)
            expert_summary.index_add_(0, dst, msg)

            degree = torch.zeros(num_diseases, device=device, dtype=msg.dtype)
            degree.index_add_(0, dst, torch.ones_like(src, dtype=msg.dtype))
            degree = degree.unsqueeze(-1).clamp(min=1.0)

            expert_summary = expert_summary / degree
            total_expert_context += alpha[:, i].unsqueeze(-1) * expert_summary

        return total_expert_context, alpha


def info_nce_loss(view1, view2, temperature=0.1):
    view1 = F.normalize(view1, dim=1)
    view2 = F.normalize(view2, dim=1)
    logits = torch.matmul(view1, view2.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def unified_loss(similarity_score, labels, view1_A, view2_A, view1_B, view2_B, beta=0.1):
    l_supervised = F.binary_cross_entropy_with_logits(similarity_score.squeeze(), labels.float())

    l_contrastive = 0.0
    if beta > 0:
        l_contrastive_A = info_nce_loss(view1_A, view2_A)
        l_contrastive_B = info_nce_loss(view1_B, view2_B)
        l_contrastive = (l_contrastive_A + l_contrastive_B) / 2

    l_total = l_supervised + beta * l_contrastive
    return l_total, l_supervised, l_contrastive
