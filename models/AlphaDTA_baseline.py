import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from models.IGN import ModifiedAttentiveFPPredictorV2, DTIConvGraph3Layer, EdgeWeightedSumAndMax, FC


class SimpleSingleExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.eps = 1e-8
        
        # Step 1: Projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Step 2: Simple Attention for pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Step 3: Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, single_emb: torch.Tensor, token_length: torch.Tensor) -> torch.Tensor:
        B, T, _ = single_emb.shape
        device = single_emb.device
        
        # Step 1: Project
        x = self.projection(single_emb)  # (B, T, hidden_dim)
        
        # Step 2: Attention pooling
        # Create mask
        mask = torch.arange(T, device=device).unsqueeze(0) < token_length.unsqueeze(1)  # (B, T)
        
        # Compute attention scores
        attn_scores = self.attention(x).squeeze(-1)  # (B, T)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax and weighted sum
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden_dim)
        
        # Step 3: Output MLP
        output = self.output_mlp(pooled)
        
        return output


class SimplePairExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.eps = 1e-8
        
        # Step 1: Projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Step 2: Simple Attention for pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Step 3: Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, pair_emb: torch.Tensor, 
                protein_length: torch.Tensor,
                token_length: torch.Tensor) -> torch.Tensor:
        B, T, _, D = pair_emb.shape
        device = pair_emb.device
        
        indices = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        protein_mask = indices < protein_length.unsqueeze(1)  # (B, T)
        ligand_mask = (indices >= protein_length.unsqueeze(1)) & (indices < token_length.unsqueeze(1))
        
        # Interaction mask: protein-ligand pairs only
        interaction_mask = (protein_mask.unsqueeze(2) & ligand_mask.unsqueeze(1)) | \
                          (ligand_mask.unsqueeze(2) & protein_mask.unsqueeze(1))  # (B, T, T)
        
        # Step 2: Project
        x = self.projection(pair_emb)  # (B, T, T, hidden_dim)
        
        # Step 3: Flatten for pooling
        x_flat = x.flatten(1, 2)  # (B, T*T, hidden_dim)
        mask_flat = interaction_mask.flatten(1, 2)  # (B, T*T)
        
        # Step 4: Attention pooling ONLY (consistent with single)
        attn_scores = self.attention(x_flat).squeeze(-1)  # (B, T*T)
        attn_scores = attn_scores.masked_fill(~mask_flat, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        pooled = (x_flat * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden_dim)
        
        # Step 5: Output MLP
        output = self.output_mlp(pooled)
        
        return output


class TripleAdaptiveFusion(nn.Module):
    """
    Exactly same as AlphaDTA
    """
    def __init__(self, ign_dim: int, single_dim: int, pair_dim: int,
                 output_dim: int, init_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)):
        super().__init__()
        
        # Validate and normalize init weights
        init_w = torch.tensor(init_weights, dtype=torch.float32)
        if (init_w <= 0).any():
            init_w = torch.full((3,), 1.0 / 3.0)
        init_w = init_w / init_w.sum()
        
        # Learnable logits
        self.fusion_logits = nn.Parameter(torch.log(init_w + 1e-8))
        
        # Projections (only if dimensions differ)
        self.ign_proj = nn.Linear(ign_dim, output_dim) if ign_dim != output_dim else nn.Identity()
        self.single_proj = nn.Linear(single_dim, output_dim) if single_dim != output_dim else nn.Identity()
        self.pair_proj = nn.Linear(pair_dim, output_dim) if pair_dim != output_dim else nn.Identity()
    
    def get_weights(self) -> Tuple[float, float, float]:
        with torch.no_grad():
            weights = F.softmax(self.fusion_logits, dim=0)
        return tuple(weights.cpu().tolist())
    
    def forward(self, ign_feat: torch.Tensor, single_feat: torch.Tensor,
                pair_feat: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.fusion_logits, dim=0)
        
        ign = self.ign_proj(ign_feat)
        single = self.single_proj(single_feat)
        pair = self.pair_proj(pair_feat)
        
        return weights[0] * ign + weights[1] * single + weights[2] * pair


class AlphaDTA_baseline(nn.Module):
    def __init__(self, ign_config: dict, emb_encoder_config: dict,
                 fusion_output_dim: int = 256,
                 init_ign_weight: float = 0.4,
                 fusion_dropout: float = 0.2):
        super().__init__()
        
        hidden_dim = emb_encoder_config['hidden_dim']

        # IGN Graph Encoders
        self.ign_cov_graph = ModifiedAttentiveFPPredictorV2(
            node_feat_size=ign_config['node_feat_size'],
            edge_feat_size=ign_config['edge_feat_size'],
            num_layers=ign_config['num_layers'],
            graph_feat_size=ign_config['graph_feat_size'],
            dropout=ign_config['dropout']
        )
        
        self.ign_noncov_graph = DTIConvGraph3Layer(
            in_dim=ign_config['graph_feat_size'] + 1,
            out_dim=ign_config['outdim_g3'],
            dropout=ign_config['dropout']
        )
        
        self.ign_readout = EdgeWeightedSumAndMax(in_feats=ign_config['outdim_g3'])
        
        ign_feature_dim = ign_config['outdim_g3'] * 2
        
        self.single_extractor = SimpleSingleExtractor(
            input_dim=emb_encoder_config['single_in_dim'],
            hidden_dim=hidden_dim,
            dropout=emb_encoder_config['dropout']
        )
        
        self.pair_extractor = SimplePairExtractor(
            input_dim=emb_encoder_config['pair_in_dim'],
            hidden_dim=hidden_dim,
            dropout=emb_encoder_config['dropout']
        )
        
        if fusion_output_dim != ign_feature_dim:
            fusion_output_dim = ign_feature_dim
        
        af3_weight = 1.0 - init_ign_weight
        init_single_weight = af3_weight * 0.5
        init_pair_weight = af3_weight * 0.5
        
        self.fusion = TripleAdaptiveFusion(
            ign_dim=ign_feature_dim,
            single_dim=hidden_dim,
            pair_dim=hidden_dim,
            output_dim=fusion_output_dim,
            init_weights=(init_ign_weight, init_single_weight, init_pair_weight)
        )
        
        self.predictor = FC(
            d_graph_layer=fusion_output_dim,
            d_FC_layer=ign_config['d_FC_layer'],
            n_FC_layer=ign_config['n_FC_layer'],
            dropout=ign_config['dropout'],
            n_tasks=ign_config['n_tasks']
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        if hasattr(self.predictor, 'predict'):
            for layer in self.predictor.predict:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 6.0)
    
    def forward(self, bg, bg3, single_emb: torch.Tensor, pair_emb: torch.Tensor,
                token_length: torch.Tensor, protein_length: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            bg: DGL graph (covalent)
            bg3: DGL graph (non-covalent)
            single_emb: (B, T, single_in_dim)
            pair_emb: (B, T, T, pair_in_dim)
            token_length: (B,)
            protein_length: (B,)
            
        Returns:
            affinity: (B,)
        """
        # === IGN Branch (IDENTICAL) ===
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        
        atom_feats = self.ign_cov_graph(bg, atom_feats, bond_feats)
        
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.ign_noncov_graph(bg3, atom_feats, bond_feats3)
        
        ign_features = self.ign_readout(bg3, bond_feats3)
        
        # === Simple AF3 Processing ===
        single_feat = self.single_extractor(single_emb, token_length)
        pair_feat = self.pair_extractor(pair_emb, protein_length, token_length)
        
        # === Fusion (IDENTICAL) ===
        fused_features = self.fusion(ign_features, single_feat, pair_feat)
        
        # === Prediction (IDENTICAL) ===
        affinity = self.predictor(fused_features).squeeze(-1)
        
        return affinity
    
    def get_fusion_info(self) -> dict:
        ign_w, single_w, pair_w = self.fusion.get_weights()
        return {
            'strategy': 'triple_adaptive',
            'ign_weight': ign_w,
            'single_weight': single_w,
            'pair_weight': pair_w,
            'af3_total_weight': single_w + pair_w
        }


def create_AlphaDTA_baseline(
    # Fusion settings
    fusion_output_dim: int = 256,
    init_ign_weight: float = 0.4,
    fusion_dropout: float = 0.2,
    
    # Simple AF3 settings
    emb_encoder_single_dim: int = 384,
    emb_encoder_pair_dim: int = 128,
    emb_encoder_hidden_dim: int = 128,
    emb_encoder_dropout: float = 0.15,
    
    # IGN settings
    graph_node_feat_size: int = 40,
    graph_edge_feat_size: int = 21,
    graph_hidden_dim: int = 128,
    graph_num_layers: int = 2,
    graph_dropout: float = 0.35,
    
    # FC settings
    fc_hidden_dim: int = 128,
    fc_num_layers: int = 2
) -> AlphaDTA_baseline:
    """Factory function for AlphaDTA_baseline."""
    
    # Auto-adjust fusion_output_dim
    expected_dim = graph_hidden_dim * 2
    if fusion_output_dim != expected_dim:
        print(f"⚠️ Auto-adjusting fusion_output_dim: {fusion_output_dim} → {expected_dim}")
        fusion_output_dim = expected_dim
    
    ign_config = {
        'node_feat_size': graph_node_feat_size,
        'edge_feat_size': graph_edge_feat_size,
        'num_layers': graph_num_layers,
        'graph_feat_size': graph_hidden_dim,
        'outdim_g3': graph_hidden_dim,
        'dropout': graph_dropout,
        'd_FC_layer': fc_hidden_dim,
        'n_FC_layer': fc_num_layers,
        'n_tasks': 1
    }
    
    emb_encoder_config = {
        'single_in_dim': emb_encoder_single_dim,
        'pair_in_dim': emb_encoder_pair_dim,
        'hidden_dim': emb_encoder_hidden_dim,
        'dropout': emb_encoder_dropout
    }
    
    return AlphaDTA_baseline(
        ign_config=ign_config,
        emb_encoder_config=emb_encoder_config,
        fusion_output_dim=fusion_output_dim,
        init_ign_weight=init_ign_weight,
        fusion_dropout=fusion_dropout
    )
