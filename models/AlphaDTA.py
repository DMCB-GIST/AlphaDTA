import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
from models.IGN import ModifiedAttentiveFPPredictorV2, DTIConvGraph3Layer, EdgeWeightedSumAndMax, FC


class ResidualTokenProjection(nn.Module):
    """
    Dimensional Adaptation + Residual Projection
    """
    def __init__(self, in_dim: int, bottleneck_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        
        # Pre-Norm
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, out_dim),
            nn.Dropout(dropout)
        )
        
        # Residual projection if dimensions differ
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm style
        out = self.net(self.norm(x))
        return self.residual_proj(x) + out


class GatedMultiScalePooling(nn.Module):
    """
    Multi-Scale Pooling: Attention + Max + Mean with gated combination.
    """
    def __init__(self, embed_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.eps = 1e-8
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1)
        )
        # Gated combination
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
            nn.Softmax(dim=-1)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        B, N, D = x.shape
        
        # Attention pooling
        attn_scores = self.attention(x).squeeze(-1)  # (B, N)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        # Max pooling
        x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        max_pooled = x_masked.max(dim=1)[0]
        max_pooled = torch.nan_to_num(max_pooled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Mean pooling
        masked_sum = (x * mask.unsqueeze(-1)).sum(dim=1)
        num_valid = mask.sum(dim=1, keepdim=True).float().clamp(min=self.eps)
        mean_pooled = masked_sum / num_valid
        
        # Gated combination
        combined = torch.cat([attn_pooled, max_pooled, mean_pooled], dim=-1)
        gates = self.gate(combined)  # (B, 3)
        
        stacked = torch.stack([attn_pooled, max_pooled, mean_pooled], dim=2)  # (B, D, 3)
        weighted = (stacked * gates.unsqueeze(1)).sum(dim=2)  # (B, D)
        return self.out_proj(weighted)


class MultiScalePooling2D(nn.Module):
    """
    2D Multi-Scale Pooling for pair embeddings.
    """
    def __init__(self, pair_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.eps = 1e-8
        
        # Importance scoring for attention pooling
        self.importance = nn.Sequential(
            nn.Linear(pair_dim, pair_dim // 2),
            nn.LayerNorm(pair_dim // 2),
            nn.ReLU(),
            nn.Linear(pair_dim // 2, 1)
        )
        
        # Single combined projection
        self.combine = nn.Sequential(
            nn.Linear(pair_dim * 3, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, pair_emb: torch.Tensor, interaction_mask: torch.Tensor) -> torch.Tensor:
        B, T, _, D = pair_emb.shape
        
        # Flatten
        pair_flat = pair_emb.flatten(1, 2)  # (B, T*T, D)
        mask_flat = interaction_mask.flatten(1, 2)  # (B, T*T)
        
        # Attention pooling
        scores = self.importance(pair_flat).squeeze(-1)
        scores = scores.masked_fill(~mask_flat, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        attn_pooled = (pair_flat * weights.unsqueeze(-1)).sum(dim=1)
        
        # Max pooling
        pair_masked = pair_flat.masked_fill(~mask_flat.unsqueeze(-1), float('-inf'))
        max_pooled = pair_masked.max(dim=1)[0]
        max_pooled = torch.nan_to_num(max_pooled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Mean pooling
        masked_sum = (pair_flat * mask_flat.unsqueeze(-1)).sum(dim=1)
        num_valid = mask_flat.sum(dim=1, keepdim=True).float().clamp(min=self.eps)
        mean_pooled = masked_sum / num_valid
        
        # Single projection
        combined = torch.cat([attn_pooled, max_pooled, mean_pooled], dim=-1)
        return self.combine(combined)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, 
                 ff_mult: int = 4, dropout: float = 0.15):
        super().__init__()
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # FFN
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_mult, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-Attention with Pre-Norm
        normed = self.norm1(x)
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout1(attn_out)
        
        # FFN with Pre-Norm
        x = x + self.ffn(self.norm2(x))
        
        return x


class CrossAttentionBlock(nn.Module):
    """
    Bidirectional Cross-Attention with separate output projections.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Pre-Norm
        self.norm_protein = nn.LayerNorm(embed_dim)
        self.norm_ligand = nn.LayerNorm(embed_dim)
        
        # QKV projections
        self.protein_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.ligand_qkv = nn.Linear(embed_dim, embed_dim * 3)
        
        # Separate output projections
        self.protein_out_proj = nn.Linear(embed_dim, embed_dim)
        self.ligand_out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        return torch.matmul(attn, v)

    def forward(self,
                protein: torch.Tensor, ligand: torch.Tensor,
                protein_mask: Optional[torch.Tensor] = None,
                ligand_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P, _ = protein.shape
        _, L, _ = ligand.shape

        # Pre-Norm
        protein_norm = self.norm_protein(protein)
        ligand_norm = self.norm_ligand(ligand)

        # QKV
        p_q, p_k, p_v = self.protein_qkv(protein_norm).chunk(3, dim=-1)
        l_q, l_k, l_v = self.ligand_qkv(ligand_norm).chunk(3, dim=-1)

        # Reshape for multi-head attention
        p_q, p_k, p_v = map(self._reshape, (p_q, p_k, p_v))
        l_q, l_k, l_v = map(self._reshape, (l_q, l_k, l_v))

        # Cross-attention masks
        if protein_mask is not None and ligand_mask is not None:
            p2l_mask = protein_mask.unsqueeze(2) & ligand_mask.unsqueeze(1)
            l2p_mask = ligand_mask.unsqueeze(2) & protein_mask.unsqueeze(1)
        else:
            p2l_mask = l2p_mask = None

        # Bidirectional cross-attention
        protein_attn = self._attention(p_q, l_k, l_v, p2l_mask)
        ligand_attn = self._attention(l_q, p_k, p_v, l2p_mask)

        # Reshape back
        protein_attn = protein_attn.transpose(1, 2).contiguous().view(B, P, self.embed_dim)
        ligand_attn = ligand_attn.transpose(1, 2).contiguous().view(B, L, self.embed_dim)

        # Separate output projections
        protein_out = self.protein_out_proj(protein_attn)
        ligand_out = self.ligand_out_proj(ligand_attn)

        return protein_out, ligand_out


class PairExtractor(nn.Module):
    """Pair embedding extractor"""
    def __init__(self, pair_dim: int, hidden_dim: int, dropout: float = 0.15, symmetrize: bool = True):
        super().__init__()
        self.symmetrize = symmetrize
        self.pooling = MultiScalePooling2D(pair_dim, hidden_dim, dropout)

    def forward(self, pair_emb: torch.Tensor,
                protein_length: torch.Tensor,
                token_length: torch.Tensor) -> torch.Tensor:
        
        B, T, _, D = pair_emb.shape
        device = pair_emb.device

        # Vectorized mask creation
        indices = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        protein_mask = indices < protein_length.unsqueeze(1)  # (B, T)
        ligand_mask = (indices >= protein_length.unsqueeze(1)) & (indices < token_length.unsqueeze(1))

        # Interaction mask: protein-ligand pairs only
        interaction_mask = (protein_mask.unsqueeze(2) & ligand_mask.unsqueeze(1)) | \
                          (ligand_mask.unsqueeze(2) & protein_mask.unsqueeze(1))

        # Symmetrize pair embeddings
        if self.symmetrize:
            pair_emb = (pair_emb + pair_emb.transpose(1, 2)) / 2.0

        return self.pooling(pair_emb, interaction_mask)


class DeepSingleExtractor(nn.Module):
    """
    Deep path: Protein/Ligand Transformer blocks + Cross-attention.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, 
                 ff_mult: int = 4, dropout: float = 0.2,
                 num_protein_layers: int = 2, num_ligand_layers: int = 3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Protein Transformer blocks
        self.protein_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_mult, dropout)
            for _ in range(num_protein_layers)
        ])
        
        # Ligand Transformer blocks
        self.ligand_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_mult, dropout)
            for _ in range(num_ligand_layers)
        ])
        
        # Cross-attention
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        
        # Pooling
        self.protein_pool = GatedMultiScalePooling(hidden_dim, hidden_dim, dropout)
        self.ligand_pool = GatedMultiScalePooling(hidden_dim, hidden_dim, dropout)
        
        # Final layer norms
        self.protein_final_norm = nn.LayerNorm(hidden_dim)
        self.ligand_final_norm = nn.LayerNorm(hidden_dim)

    def _extract_tokens_vectorized(self, single_emb: torch.Tensor,
                                    protein_length: torch.Tensor,
                                    token_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                                          torch.Tensor, torch.Tensor]:
        B, T, D = single_emb.shape
        device = single_emb.device

        # Calculate max lengths
        max_p_len = protein_length.max().int().item()
        max_l_len = (token_length - protein_length).max().int().item()
        max_p_len = max(max_p_len, 1)
        max_l_len = max(max_l_len, 1)

        # Create index tensors
        p_indices = torch.arange(max_p_len, device=device).unsqueeze(0)  # (1, max_p_len)
        l_indices = torch.arange(max_l_len, device=device).unsqueeze(0)  # (1, max_l_len)

        # Masks
        protein_mask = p_indices < protein_length.unsqueeze(1)  # (B, max_p_len)
        ligand_lengths = token_length - protein_length
        ligand_mask = l_indices < ligand_lengths.unsqueeze(1)  # (B, max_l_len)

        # Initialize output tensors
        protein_tokens = torch.zeros(B, max_p_len, D, device=device, dtype=single_emb.dtype)
        ligand_tokens = torch.zeros(B, max_l_len, D, device=device, dtype=single_emb.dtype)

        # Vectorized extraction using advanced indexing
        # For protein tokens: indices 0 to protein_length-1
        for i in range(B):
            p_len = protein_length[i]
            l_start = p_len
            l_end = token_length[i]
            
            protein_tokens[i, :p_len] = single_emb[i, :p_len]
            if l_end > l_start:
                ligand_tokens[i, :(l_end - l_start)] = single_emb[i, l_start:l_end]

        return protein_tokens, ligand_tokens, protein_mask, ligand_mask

    def forward(self, single_emb: torch.Tensor,
                protein_length: torch.Tensor,
                token_length: torch.Tensor) -> torch.Tensor:
        
        # Extract tokens
        protein, ligand, p_mask, l_mask = self._extract_tokens_vectorized(
            single_emb, protein_length, token_length
        )

        # Protein Transformer blocks
        for block in self.protein_blocks:
            protein = block(protein, p_mask)

        # Ligand Transformer blocks
        for block in self.ligand_blocks:
            ligand = block(ligand, l_mask)

        # Cross-attention
        p_cross, l_cross = self.cross_attn(protein, ligand, p_mask, l_mask)
        protein = protein + p_cross
        ligand = ligand + l_cross

        # Final normalization
        protein = self.protein_final_norm(protein)
        ligand = self.ligand_final_norm(ligand)

        # Pooling
        protein_feat = self.protein_pool(protein, p_mask)
        ligand_feat = self.ligand_pool(ligand, l_mask)

        return torch.cat([protein_feat, ligand_feat], dim=-1)


class ShallowSingleExtractor(nn.Module):
    """
    Shallow path: Single Transformer block for efficiency.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, 
                 ff_mult: int = 4, dropout: float = 0.15):
        super().__init__()
        
        # Single Transformer block
        self.block = TransformerBlock(hidden_dim, num_heads, ff_mult, dropout)
        
        # Final norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Pooling
        self.pooling = GatedMultiScalePooling(hidden_dim, hidden_dim, dropout)

    def forward(self, single_emb: torch.Tensor, token_length: torch.Tensor) -> torch.Tensor:
        B, T, _ = single_emb.shape
        device = single_emb.device

        # Create mask
        mask = torch.arange(T, device=device).unsqueeze(0) < token_length.unsqueeze(1)

        # Transformer block
        x = self.block(single_emb, mask)
        
        # Final norm
        x = self.final_norm(x)

        # Pooling
        return self.pooling(x, mask)


class TripleAdaptiveFusion(nn.Module):
    """
    3-way adaptive fusion: IGN / Deep / Shallow.
    Softmax-normalized learnable weights.
    """
    def __init__(self, ign_dim: int, deep_dim: int, shallow_dim: int,
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
        self.deep_proj = nn.Linear(deep_dim, output_dim) if deep_dim != output_dim else nn.Identity()
        self.shallow_proj = nn.Linear(shallow_dim, output_dim) if shallow_dim != output_dim else nn.Identity()

    def get_weights(self) -> Tuple[float, float, float]:
        with torch.no_grad():
            weights = F.softmax(self.fusion_logits, dim=0)
        return tuple(weights.cpu().tolist())

    def forward(self, ign_feat: torch.Tensor, deep_feat: torch.Tensor,
                shallow_feat: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.fusion_logits, dim=0)
        
        ign = self.ign_proj(ign_feat)
        deep = self.deep_proj(deep_feat)
        shallow = self.shallow_proj(shallow_feat)
        
        return weights[0] * ign + weights[1] * deep + weights[2] * shallow


class AlphaDTA(nn.Module):
    def __init__(self, ign_config: dict, emb_encoder_config: dict,
                 fusion_output_dim: int = 256,
                 init_ign_weight: float = 0.4,
                 fusion_dropout: float = 0.2):
        super().__init__()
        
        hidden_dim = emb_encoder_config['hidden_dim']
        ff_mult = emb_encoder_config.get('ff_mult', 4)
        
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
        
        # Shared Single Projection
        self.single_proj = ResidualTokenProjection(
            in_dim=emb_encoder_config['single_in_dim'],
            bottleneck_dim=emb_encoder_config['single_in_dim'] // 2,
            out_dim=hidden_dim,
            dropout=emb_encoder_config['dropout'] * 0.5
        )
        
        # Shared Pair Extractor
        self.pair_extractor = PairExtractor(
            pair_dim=emb_encoder_config['pair_in_dim'],
            hidden_dim=hidden_dim,
            dropout=emb_encoder_config['dropout'],
            symmetrize=emb_encoder_config.get('symmetrize_pair', True)
        )
        
        # Deep Path
        self.deep_extractor = DeepSingleExtractor(
            hidden_dim=hidden_dim,
            num_heads=emb_encoder_config['num_heads'],
            ff_mult=ff_mult,
            dropout=emb_encoder_config['dropout'],
            num_protein_layers=emb_encoder_config['num_protein_layers'],
            num_ligand_layers=emb_encoder_config['num_ligand_layers']
        )
        
        # Deep fusion: [protein, ligand, pair] -> hidden_dim
        self.deep_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(fusion_dropout)
        )
        
        # Shallow Path
        self.shallow_extractor = ShallowSingleExtractor(
            hidden_dim=hidden_dim,
            num_heads=emb_encoder_config['num_heads'],
            ff_mult=ff_mult,
            dropout=emb_encoder_config['dropout']
        )
        
        # Shallow fusion: [single, pair] -> hidden_dim
        self.shallow_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout)
        )
        
        # Triple Adaptive Fusion
        if fusion_output_dim != ign_feature_dim:
            fusion_output_dim = ign_feature_dim
        
        emb_encoder_weight = 1.0 - init_ign_weight
        init_deep_weight = emb_encoder_weight * 0.5
        init_shallow_weight = emb_encoder_weight * 0.5
        
        self.fusion = TripleAdaptiveFusion(
            ign_dim=ign_feature_dim,
            deep_dim=hidden_dim,
            shallow_dim=hidden_dim,
            output_dim=fusion_output_dim,
            init_weights=(init_ign_weight, init_deep_weight, init_shallow_weight)
        )
        
        # Final Predictor
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
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)
        
        # Smaller init for final prediction layer
        if hasattr(self.predictor, 'predict'):
            for layer in self.predictor.predict:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 6.0)  # Target mean

    def forward(self, bg, bg3, single_emb: torch.Tensor, pair_emb: torch.Tensor,
                token_length: torch.Tensor, protein_length: torch.Tensor) -> torch.Tensor:
        """
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
        # === IGN Path ===
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        
        atom_feats = self.ign_cov_graph(bg, atom_feats, bond_feats)
        
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.ign_noncov_graph(bg3, atom_feats, bond_feats3)
        
        ign_features = self.ign_readout(bg3, bond_feats3)
        
        # === Shared Processing ===
        single_proj = self.single_proj(single_emb)
        pair_feat = self.pair_extractor(pair_emb, protein_length, token_length)
        
        # === Deep Path ===
        deep_single_feat = self.deep_extractor(single_proj, protein_length, token_length)
        deep_feat = self.deep_fusion(torch.cat([deep_single_feat, pair_feat], dim=-1))
        
        # === Shallow Path ===
        shallow_single_feat = self.shallow_extractor(single_proj, token_length)
        shallow_feat = self.shallow_fusion(torch.cat([shallow_single_feat, pair_feat], dim=-1))
        
        # === Fusion ===
        fused_features = self.fusion(ign_features, deep_feat, shallow_feat)
        
        # === Prediction ===
        affinity = self.predictor(fused_features).squeeze(-1)
        
        return affinity

    def get_fusion_info(self) -> dict:
        """Get fusion weights and info."""
        ign_w, deep_w, shallow_w = self.fusion.get_weights()
        return {
            'strategy': 'triple_adaptive',
            'ign_weight': ign_w,
            'deep_weight': deep_w,
            'shallow_weight': shallow_w,
            'emb_encoder_total_weight': deep_w + shallow_w
        }


def create_AlphaDTA(
    # Fusion settings
    fusion_output_dim: int = 256,
    init_ign_weight: float = 0.4,
    fusion_dropout: float = 0.2,
    
    # Embedding Encoder settings
    emb_encoder_single_dim: int = 384,
    emb_encoder_pair_dim: int = 128,
    emb_encoder_hidden_dim: int = 128,
    emb_encoder_num_heads: int = 4,
    emb_encoder_ff_mult: int = 4,
    emb_encoder_num_protein_layers: int = 2,
    emb_encoder_num_ligand_layers: int = 3,
    emb_encoder_dropout: float = 0.25,
    
    # IGN settings
    graph_node_feat_size: int = 40,
    graph_edge_feat_size: int = 21,
    graph_hidden_dim: int = 128,
    graph_num_layers: int = 2,
    graph_dropout: float = 0.35,
    
    # FC settings
    fc_hidden_dim: int = 128,
    fc_num_layers: int = 2
) -> AlphaDTA:
    
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
        'num_heads': emb_encoder_num_heads,
        'ff_mult': emb_encoder_ff_mult,
        'num_protein_layers': emb_encoder_num_protein_layers,
        'num_ligand_layers': emb_encoder_num_ligand_layers,
        'dropout': emb_encoder_dropout,
        'symmetrize_pair': True
    }
    
    return AlphaDTA(
        ign_config=ign_config,
        emb_encoder_config=emb_encoder_config,
        fusion_output_dim=fusion_output_dim,
        init_ign_weight=init_ign_weight,
        fusion_dropout=fusion_dropout
    )
