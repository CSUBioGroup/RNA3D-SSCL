import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint
from typing import Dict,Tuple, Optional
class LinearProjector(nn.Module):  
    def __init__(self, input_dim, output_dim):  
        super().__init__()  
        self.linear = nn.Linear(input_dim, output_dim)  

    def forward(self, inputs):  
        return self.linear(inputs)

class BiasFreeLinearProjector(nn.Module):  
    def __init__(self, input_size, output_size):  
        super().__init__() 
        self.linear = nn.Linear(input_size, output_size, bias=False)  

    def forward(self, inputs):  
        return self.linear(inputs)




def transform_molecular_atoms(atoms, rotations, translations): 
    return torch.einsum('bja,bad->bjd', atoms, rotations) + translations[:, None, :]



def transform_ipa_data(ipa_tensor, rotations, translations):  
    return torch.einsum('bija,bad->bijd', ipa_tensor, rotations) + translations[:, None, None, :]

def inverse_transform_ipa_data(tensor, rotations, translations):  
    shifted = tensor - translations[:, None, None, :]
    return torch.einsum('bija,bad->bijd', shifted, rotations.transpose(-1, -2))  



def update_transformation_matrix(matrix, vector, rotations, translations):  # 函数名修改
    transformed_matrix = torch.einsum('bja,bad->bjd', matrix, rotations)
    transformed_vector = torch.einsum('ba,bad->bd', vector, rotations) + translations
    return transformed_matrix, transformed_vector

def quat2rot(q, L):
    scale = ((q ** 2).sum(dim=-1, keepdim=True) + 1)[:, :, None]
    u = torch.empty([L, 3, 3], device=q.device)
    u[:, 0, 0] = 1 * 1 + q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2]
    u[:, 0, 1] = 2 * (q[:, 0] * q[:, 1] - 1 * q[:, 2])
    u[:, 0, 2] = 2 * (q[:, 0] * q[:, 2] + 1 * q[:, 1])
    u[:, 1, 0] = 2 * (q[:, 0] * q[:, 1] + 1 * q[:, 2])
    u[:, 1, 1] = 1 * 1 - q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2]
    u[:, 1, 2] = 2 * (q[:, 1] * q[:, 2] - 1 * q[:, 0])
    u[:, 2, 0] = 2 * (q[:, 0] * q[:, 2] - 1 * q[:, 1])
    u[:, 2, 1] = 2 * (q[:, 1] * q[:, 2] + 1 * q[:, 0])
    u[:, 2, 2] = 1 * 1 - q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2]
    return u / scale




class MSAFeaturesAttention(nn.Module):
    def __init__(self, msa_dim, pair_dim, num_heads=8, head_dim=8):
        super().__init__()
        self.N_head = num_heads
        self.c = head_dim
        self.sq_c = 1 / math.sqrt(head_dim)
        self.norm1 = nn.LayerNorm(msa_dim)
        self.qlinear = BiasFreeLinearProjector(msa_dim, num_heads * head_dim)
        self.klinear = BiasFreeLinearProjector(msa_dim, num_heads * head_dim)
        self.vlinear = BiasFreeLinearProjector(msa_dim, num_heads * head_dim)
        self.norm_z = nn.LayerNorm(pair_dim)
        self.zlinear = BiasFreeLinearProjector(pair_dim, num_heads)
        self.glinear = LinearProjector(msa_dim, num_heads * head_dim)
        self.olinear = LinearProjector(num_heads * head_dim, msa_dim)

    def forward(self, msa_input, pair_input):
        batch_size, seq_len, embed_dim = msa_input.shape
        normed_msa = self.norm1(msa_input)
        q_proj = self.qlinear(normed_msa).reshape(batch_size, seq_len, self.N_head, self.c)
        k_proj = self.klinear(normed_msa).reshape(batch_size, seq_len, self.N_head, self.c)
        v_proj = self.vlinear(normed_msa).reshape(batch_size, seq_len, self.N_head, self.c)
        bias = self.zlinear(self.norm_z(pair_input))
        gate = torch.sigmoid(self.glinear(normed_msa)).reshape(batch_size, seq_len, self.N_head, self.c)
        attention_scores = torch.einsum('bqhc,bvhc->bqvh', q_proj, k_proj) * (self.sq_c) + bias[None, :, :, :]
        attention_probs = F.softmax(attention_scores, dim=2)
        weighted_values = torch.einsum('bqvh,bvhc->bqhc', attention_probs, v_proj) * gate
        output = self.olinear(weighted_values.reshape(batch_size, seq_len, -1))
        return output


class SequenceFeaturesAttention(nn.Module):
    def __init__(self, msa_dim, num_heads=8, head_dim=8):
        super().__init__()
        self.N_head = num_heads
        self.c = head_dim
        self.sq_c = 1 / math.sqrt(head_dim)
        self.norm1 = nn.LayerNorm(msa_dim)
        self.qlinear = BiasFreeLinearProjector(msa_dim, num_heads * head_dim)
        self.klinear = BiasFreeLinearProjector(msa_dim, num_heads * head_dim)
        self.vlinear = BiasFreeLinearProjector(msa_dim, num_heads * head_dim)
        self.glinear = LinearProjector(msa_dim, num_heads * head_dim)
        self.olinear = LinearProjector(num_heads * head_dim, msa_dim)

    def forward(self, msa_input):
        # msa_input : N L 32
        batch_size, seq_len, embed_dim = msa_input.shape
        normed_msa = self.norm1(msa_input)
        q_proj = self.qlinear(normed_msa).reshape(batch_size, seq_len, self.N_head, self.c)
        k_proj = self.klinear(normed_msa).reshape(batch_size, seq_len, self.N_head, self.c)
        v_proj = self.vlinear(normed_msa).reshape(batch_size, seq_len, self.N_head, self.c)
        gate = torch.sigmoid(self.glinear(normed_msa)).reshape(batch_size, seq_len, self.N_head, self.c)
        attention_scores = torch.einsum('slhc,tlhc->stlh', q_proj, k_proj) * (self.sq_c)
        attention_probs = F.softmax(attention_scores, dim=1)
        weighted_values = torch.einsum('stlh,tlhc->slhc', attention_probs, v_proj) * gate
        output = self.olinear(weighted_values.reshape(batch_size, seq_len, -1))
        return output



class MSATransition(nn.Module):
    def __init__(self, input_dim, expand_ratio=2):
        super().__init__()
        self.c_expand = 4
        self.m_dim = input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.linear1 = LinearProjector(input_dim, input_dim * expand_ratio)
        self.linear2 = LinearProjector(input_dim * expand_ratio, input_dim)

    def forward(self, msa_tensor):
        normed = self.norm(msa_tensor)
        expanded = self.linear1(normed)
        output = self.linear2(F.relu(expanded))
        return output


class MSAOuterProductMean(nn.Module):
    def __init__(self, msa_dim, pair_dim, head_dim=12):
        super().__init__()
        self.m_dim = msa_dim
        self.c = head_dim
        self.norm = nn.LayerNorm(msa_dim)
        self.linear1 = LinearProjector(msa_dim, head_dim)
        self.linear2 = LinearProjector(msa_dim, head_dim)
        self.linear3 = LinearProjector(head_dim * head_dim, pair_dim)

    def forward(self, msa_input):
        _, seq_len, _ = msa_input.shape
        normed = self.norm(msa_input)
        proj_a = self.linear2(normed)
        proj_b = self.linear1(normed)
        outer = torch.einsum('nia,njb->nijab', proj_a, proj_b).mean(dim=0)
        output = self.linear3(outer.reshape(seq_len, seq_len, -1))
        return output



class MSAInitializer(nn.Module):  # Changed class name for clarity
    def __init__(self, sequence_dim, msa_input_dim, msa_embedding_dim, pairwise_dim):
        super().__init__()
        self.msalinear = LinearProjector(msa_input_dim, msa_embedding_dim)
        self.qlinear = LinearProjector(sequence_dim, pairwise_dim)
        self.klinear = LinearProjector(sequence_dim, pairwise_dim)
        self.slinear = LinearProjector(sequence_dim, msa_embedding_dim)
        self.pos = self._generate_pairwise_positional_encoding().float()  # Changed method name
        self.pos1d = self._generate_1d_positional_encoding()              # Changed method name
        self.poslinear = LinearProjector(65, pairwise_dim)
        self.poslinear2 = LinearProjector(14, msa_embedding_dim)

    def move_module_to_device(self, device):  # Changed method name
        self.to(device)
        self.pos.to(device)

    def _generate_1d_positional_encoding(self, max_length=2000):  # Changed method name and parameter name
        position_indices = torch.arange(max_length)
        num_bits = 14
        binary_encoding = (((position_indices[:, None] & (1 << np.arange(num_bits)))) > 0).float()
        return binary_encoding

    def _generate_pairwise_positional_encoding(self, max_length=2000):  # Changed method name and parameter name
        indices = torch.arange(max_length)
        pairwise_diff = (indices[None, :] - indices[:, None]).clamp(-32, 32)
        return F.one_hot(pairwise_diff + 32, 65)

    def forward(self, sequence_input, msa_input):  # Changed parameter names
        # Local variable names changed for clarity
        if self.pos.device != msa_input.device:
            self.pos = self.pos.to(msa_input.device)
        if self.pos1d.device != msa_input.device:
            self.pos1d = self.pos1d.to(msa_input.device)
        _, seq_length, _ = msa_input.shape
        sequence_embedding = self.slinear(sequence_input)
        msa_embedding = self.msalinear(msa_input)
        pos1d_embedding = self.poslinear2(self.pos1d[:seq_length])

        msa_embedding = msa_embedding + sequence_embedding[None, :, :] + pos1d_embedding[None, :, :]

        query_embedding = self.qlinear(sequence_input)
        key_embedding = self.klinear(sequence_input)
        pairwise_embedding = query_embedding[None, :, :] + key_embedding[:, None, :]

        pairwise_embedding = pairwise_embedding + self.poslinear(self.pos[:seq_length, :seq_length])
        return msa_embedding, pairwise_embedding


class Evoformer(nn.Module):
    def __init__(self,m_dim,z_dim,docheck=False):
        super(Evoformer,self).__init__()
        self.layers=[16]
        self.docheck=docheck
        if docheck:
            pass
            #print('will do checkpoint')
        self.evos=nn.ModuleList([EvoBlock(m_dim,z_dim,True) for i in range(self.layers[0])])

    def layerfunc(self,layermodule,m,z):
        m_,z_=layermodule(m,z)
        return m_,z_


    def forward(self,m,z):
        for i in range(self.layers[0]):
            m,z=self.evos[i](m,z)

        return m,z


def create_fourier_positional_encodings(x, num_encodings=20, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x



class RecyclingEmbedder(nn.Module):
    """RecyclingEmbedder with significant implementation differences"""
    def __init__(
        self, 
        feature_dim: int, 
        state_dim: int, 
        encoding_dim: int
    ):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.linear = LinearProjector(2 * encoding_dim + 1, state_dim)
        self.normz = nn.LayerNorm(state_dim)
        self.normm = nn.LayerNorm(feature_dim)

    def compute_positional_encodings(
        self, 
        coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Compute positional encodings from coordinates"""
        # Compute Cβ distances
        cbeta_positions = coordinates[:, -1]
        distance_matrix = (cbeta_positions[:, None, :] - cbeta_positions[None, :, :]).norm(dim=-1)
        return create_fourier_positional_encodings(distance_matrix, self.encoding_dim)

    def forward(
        self, 
        features: torch.Tensor, 
        states: torch.Tensor, 
        coordinates: torch.Tensor, 
        initial_pass: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process recycling embeddings"""
        position_encodings = self.compute_positional_encodings(coordinates)#
        positional_features = self.linear(position_encodings)
        
        if initial_pass:
            return 0, positional_features
        else:
            return (

                self.normm(features),
                self.normz(states) + positional_features
            )



class PairwiseFeatureConstructor(nn.Module):

    def __init__(
        self, 
        feature_dim: int, 
        projection_dim: int = 32
    ):
        super().__init__()
        self.projection_dim = projection_dim
        
        # Feature normalization layers
        self.norm = nn.LayerNorm(feature_dim)
        self.onorm = nn.LayerNorm(projection_dim)
        
        # Projection layers
        self.alinear = LinearProjector(feature_dim, projection_dim)
        self.blinear = LinearProjector(feature_dim, projection_dim)
        
        # Gating layers
        self.aglinear = LinearProjector(feature_dim, projection_dim)
        self.bglinear = LinearProjector(feature_dim, projection_dim)
        
        # Output processing layers
        self.olinear = LinearProjector(projection_dim, feature_dim)
        self.glinear = LinearProjector(feature_dim, feature_dim)

    def compute_pairwise_features(
        self, 
        left_proj: torch.Tensor, 
        right_proj: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise outer product features"""
        return torch.einsum('ilc,jlc->ijc', left_proj, right_proj)


    def forward(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """Construct pairwise features from input representations"""
        normalized_features = self.norm(features)
        
        # Compute gated projections
        left_proj = self.alinear(normalized_features) * torch.sigmoid(self.aglinear(normalized_features))
        right_proj = self.alinear(normalized_features) * torch.sigmoid(self.aglinear(normalized_features))
        
        # Construct and normalize pairwise features
        pairwise_features = self.compute_pairwise_features(left_proj, right_proj)
        normalized_pairwise = self.onorm(pairwise_features)
        
        # Project back to feature dimension
        projected_features = self.olinear(normalized_pairwise)
        
        # Apply output gating
        gate_signal = torch.sigmoid(self.glinear(normalized_features))
        return projected_features * gate_signal


class PairwiseFeatureInjector(nn.Module):
    def __init__(
        self, 
        feature_dim: int, 
        projection_dim: int = 32
    ):
        super().__init__()
        self.projection_dim = projection_dim
        
        # Feature normalization layers
        self.norm = nn.LayerNorm(feature_dim)
        self.onorm = nn.LayerNorm(projection_dim)
        
        # Projection layers
        self.alinear = LinearProjector(feature_dim, projection_dim)
        self.blinear = LinearProjector(feature_dim, projection_dim)
        
        # Gating layers
        self.aglinear = LinearProjector(feature_dim, projection_dim)
        self.bglinear = LinearProjector(feature_dim, projection_dim)
        
        # Output processing layers
        self.olinear = LinearProjector(projection_dim, feature_dim)
        self.glinear = LinearProjector(feature_dim, feature_dim)

    def compute_injected_features(
        self, 
        left_proj: torch.Tensor, 
        right_proj: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise inner product features"""
        return torch.einsum('lic,ljc->ijc', left_proj, right_proj)

    def forward(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """Inject pairwise features into sequence representations"""
        normalized_features = self.norm(features)
        
        # Compute gated projections
        left_proj = self.alinear(normalized_features) * torch.sigmoid(self.aglinear(normalized_features))
        right_proj = self.alinear(normalized_features) * torch.sigmoid(self.aglinear(normalized_features))
        
        # Construct and normalize pairwise features
        pairwise_features = self.compute_injected_features(left_proj, right_proj)
        normalized_pairwise = self.onorm(pairwise_features)
        
        # Project back to feature dimension
        projected_features = self.olinear(normalized_pairwise)
        
        # Apply output gating
        gate_signal = torch.sigmoid(self.glinear(normalized_features))
        return projected_features * gate_signal



class PairwiseAttentionInitial(nn.Module):
    def __init__(
        self, 
        feature_dim: int, 
        heads: int = 4, 
        head_size: int = 8
    ):
        super().__init__()
        self.head_count = heads
        self.head_size = head_size
        self.sq_c = 1 / math.sqrt(head_size)
        
        # Feature normalization
        self.norm = nn.LayerNorm(feature_dim)
        
        # Attention projection layers
        self.qlinear = LinearProjector(feature_dim, heads * head_size)
        self.klinear = LinearProjector(feature_dim, heads * head_size)
        self.vlinear = LinearProjector(feature_dim, heads * head_size)
        
        # Bias and gating layers
        self.blinear = LinearProjector(feature_dim, heads)
        self.glinear = LinearProjector(feature_dim, heads * head_size)
        
        # Output projection
        self.olinear = LinearProjector(heads * head_size, feature_dim)

    def compute_attention_weights(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        biases: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention scores and apply softmax"""
        attention_logits = torch.einsum('blhc,bkhc->blkh', queries, keys) * self.sq_c
        return F.softmax(attention_logits + biases[None, :, :, :], dim=2)

    def process_attention_output(
        self, 
        attention_output: torch.Tensor, 
        gate_signal: torch.Tensor
    ) -> torch.Tensor:
        """Apply gating and reshape attention output"""
        gated_output = attention_output * gate_signal
        return gated_output.reshape(gated_output.size(0), gated_output.size(1), -1)

    def forward(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute initial pairwise attention"""
        normalized_features = self.norm(features)
        seq_length = features.size(1)
        
        # Project features to Q/K/V
        queries = self.qlinear(normalized_features).reshape(-1, seq_length, self.head_count, self.head_size)
        keys = self.klinear(normalized_features).reshape(-1, seq_length, self.head_count, self.head_size)
        values = self.vlinear(normalized_features).reshape(-1, seq_length, self.head_count, self.head_size)
        
        # Compute attention components
        attention_biases = self.blinear(normalized_features)
        gate_signal = torch.sigmoid(self.glinear(normalized_features)).view_as(queries)
        
        # Compute attention weights and output
        weights = self.compute_attention_weights(queries, keys, attention_biases)
        attention_output = torch.einsum('blkh,bkhc->blhc', weights, values)
        
        # Process and project output
        processed_output = self.process_attention_output(attention_output, gate_signal)
        return self.olinear(processed_output)




class PairwiseAttentionTerminal(nn.Module):

    def __init__(
        self, 
        feature_dim: int, 
        heads: int = 4, 
        head_size: int = 8
    ):
        super().__init__()
        self.head_count = heads
        self.head_size = head_size
        self.sq_c = 1 / math.sqrt(head_size)
        
        # Feature normalization
        self.norm = nn.LayerNorm(feature_dim)
        
        # Attention projection layers
        self.qlinear = LinearProjector(feature_dim, heads * head_size)
        self.klinear = LinearProjector(feature_dim, heads * head_size)
        self.vlinear = LinearProjector(feature_dim, heads * head_size)
        
        # Bias and gating layers
        self.blinear = LinearProjector(feature_dim, heads)
        self.glinear = LinearProjector(feature_dim, heads * head_size)
        
        # Output projection
        self.olinear = LinearProjector(heads * head_size, feature_dim)

    def compute_attention_weights(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        biases: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention scores and apply softmax"""
        attention_logits = torch.einsum('blhc,kbhc->blkh', queries, keys) * self.sq_c
        return F.softmax(attention_logits + biases[None, :, :, :].permute(0, 2, 1, 3), dim=2)

    def process_attention_output(
        self, 
        attention_output: torch.Tensor, 
        gate_signal: torch.Tensor
    ) -> torch.Tensor:
        """Apply gating and reshape attention output"""
        gated_output = attention_output * gate_signal
        return gated_output.reshape(gated_output.size(0), gated_output.size(1), -1)

    def forward(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute terminal pairwise attention"""
        normalized_features = self.norm(features)
        seq_length = features.size(1)
        
        # Project features to Q/K/V
        queries = self.qlinear(normalized_features).reshape(-1, seq_length, self.head_count, self.head_size)
        keys = self.klinear(normalized_features).reshape(-1, seq_length, self.head_count, self.head_size)
        values = self.vlinear(normalized_features).reshape(-1, seq_length, self.head_count, self.head_size)
        
        # Compute attention components
        attention_biases = self.blinear(normalized_features)
        gate_signal = torch.sigmoid(self.glinear(normalized_features)).view_as(queries)
        
        # Compute attention weights and output
        weights = self.compute_attention_weights(queries, keys, attention_biases)
        attention_output = torch.einsum('blkh,klhc->blhc', weights, values)
        
        # Process and project output
        processed_output = self.process_attention_output(attention_output, gate_signal)
        return self.olinear(processed_output)

    def alternative_forward(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """Alternative implementation with permuted inputs"""
        # Transpose feature dimensions (batch and sequence)
        transposed_features = features.permute(1, 0, 2)
        normalized_features = self.norm(transposed_features)
        seq_length = transposed_features.size(1)
        
        # Project features to Q/K/V
        queries = self.qlinear(normalized_features).reshape(-1, seq_length, self.head_count, self.head_size)
        keys = self.klinear(normalized_features).reshape(-1, seq_length, self.head_count, self.head_size)
        values = self.vlinear(normalized_features).reshape(-1, seq_length, self.head_count, self.head_size)
        
        # Compute attention components
        attention_biases = self.blinear(normalized_features)
        gate_signal = torch.sigmoid(self.glinear(normalized_features)).view_as(queries)
        
        # Compute standard attention
        weights = F.softmax(
            torch.einsum('blhc,bkhc->blkh', queries, keys) * self.sq_c + attention_biases[None, :, :, :], 
            dim=2
        )
        attention_output = torch.einsum('blkh,bkhc->blhc', weights, values)
        
        # Process output and transpose back
        processed_output = self.process_attention_output(attention_output, gate_signal)
        projected = self.olinear(processed_output)
        return projected.permute(1, 0, 2)



class PairTransition(nn.Module):
    def __init__(
        self, 
        feature_dim: int, 
        expansion_ratio: int = 2
    ):
        super().__init__()
        self.expansion_dim = feature_dim * expansion_ratio
        
        # Feature processing layers
        self.norm = nn.LayerNorm(feature_dim)
        self.linear1 = LinearProjector(feature_dim, self.expansion_dim)
        self.linear2 = LinearProjector(self.expansion_dim, feature_dim)

    def apply_activation(
        self, 
        expanded_features: torch.Tensor
    ) -> torch.Tensor:
        """Apply ReLU activation to expanded features"""
        return F.relu(expanded_features)

    def forward(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """Transform pairwise feature representations"""
        normalized_features = self.norm(features)
        expanded_features = self.linear1(normalized_features)
        activated_features = self.apply_activation(expanded_features)
        return self.linear2(activated_features)

  
    

class EvolutionaryStructurePredictor(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        msa_dim: int,
        ensemble_count: int,
        cycle_count: int,
        feature_dim: int = 64,
        state_dim: int = 128,
        pair_dim: int = 64
    ):
        super().__init__()
        self.msa_dim = msa_dim
        self.m_dim = feature_dim
        self.z_dim = pair_dim
        self.N_cycle = cycle_count
        
        # Prediction modules
        self.msa_predor = LinearProjector(feature_dim, msa_dim - 1)
        self.dis_predor = LinearProjector(pair_dim, 38)  # 36+2
        
        # Processing modules
        self.msaxyzone = EvolutionaryCycleProcessor(
            sequence_dim, msa_dim, ensemble_count, feature_dim, state_dim, pair_dim
        )
        self.structurenet = StructureModule(state_dim, pair_dim, 4, state_dim)
        
        # Placeholder for state projections
        self.slinear = LinearProjector(feature_dim, state_dim)
        
        # Virtual parameters for code obfuscation


    def predict_structures(
        self,
        msa_data: torch.Tensor,
        ss_data: torch.Tensor,
        reference_points: torch.Tensor,
        cycle_count: int
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, ...]], torch.Tensor]:
        """Execute full prediction cycle"""
        sequence_length = msa_data.size(1)
        device = msa_data.device
        
        # Initialize placeholders
        prior_features = torch.zeros_like(msa_data, requires_grad=True)
        prior_states = torch.zeros(self.z_dim, sequence_length, device=device, requires_grad=True)
        current_points = torch.zeros(sequence_length, 3, 3, device=device, requires_grad=True)
        predictions = {}
        
        
        # Iterate through cycles
        for cycle_idx in range(cycle_count):

            prior_features, prior_states, s = self.msaxyzone.pred(msa_data, ss_data, prior_features, prior_states, current_points, cycle_idx)
            current_points, rotations, translations = self.structurenet.pred(s, prior_states, reference_points)

            predictions[cycle_idx] = (current_points, rotations, translations)

        
        # Predict final distance matrix
        distance_predictions = F.softmax(self.dis_predor(prior_states), dim=-1)
        
        return predictions, distance_predictions



class EvolutionaryCycleProcessor(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        msa_dim: int,
        ensemble_count: int,
        feature_dim: int = 64,
        state_dim: int = 128,
        pair_dim: int = 64, 
    ):
        super().__init__()
        self.msa_dim = msa_dim
        self.feature_dim = feature_dim
        self.pair_dim = pair_dim
        self.sequence_dim = sequence_dim
        self.ensemble_count = ensemble_count
        
        # Secondary structure processing
        self.pre_z = nn.Linear(4, pair_dim)
        
        # MSA processing components
        self.premsa = MSAInitializer(sequence_dim, msa_dim, feature_dim, pair_dim)
        self.re_emb = RecyclingEmbedder(feature_dim, pair_dim, encoding_dim=64)
        self.evmodel = Evoformer(feature_dim, pair_dim, True)
        self.slinear = LinearProjector(pair_dim, state_dim)

    def generate_msa_mask(self, msa_input):
        """Create masking tensor for MSA data"""
        num_rows, seq_length = msa_input.shape[:2]
        return torch.zeros(num_rows, seq_length, device=msa_input.device)

    def pred(self, msa_input,ss_input, prior_m, prior_z, coord_input, cycle_index):
        m_sum, z_sum, s_sum = 0, 0, 0
        for i in range(self.ensemble_count):
            msa_mask = self.generate_msa_mask(msa_input)
            msa_true = msa_input + 0
            seq = msa_true[0] * 1.0  # 22-dim
            msa = torch.cat([msa_true * (1 - msa_mask[:, :, None]), msa_mask[:, :, None]], dim=-1)
            
            initial_m, initial_z = self.premsa(seq, msa)
            ss_embed = self.pre_z(ss_input)
            modified_z = initial_z + ss_embed
            rec_m, state_embedding = self.re_emb(prior_m, prior_z, coord_input, cycle_index == 0)  # already added residually
            modified_z = modified_z + state_embedding
            modified_m = torch.cat([(initial_m[0] + rec_m)[None, ...], initial_m[1:]], dim=0)
            modified_m, modified_z = self.evmodel(modified_m, modified_z)
            final_s = self.slinear(modified_m[0])
            m_sum = m_sum + modified_m[0]
            z_sum = z_sum + modified_z
            s_sum = s_sum + final_s
        # Calculate ensemble averages
        divisor = float(self.ensemble_count)
        return m_sum / divisor, z_sum / divisor, s_sum / divisor   





class InvariantPointAttention(nn.Module):
    def __init__(self,dim_in,dim_z,N_head=8,c=16,N_query=4,N_p_values=6,) -> None:
        super(InvariantPointAttention,self).__init__()
        self.dim_in=dim_in
        self.dim_z=dim_z
        self.N_head =N_head
        self.c=c
        self.c_squ = 1.0/math.sqrt(c)
        self.W_c = math.sqrt(2.0/(9*N_query))
        self.W_L = math.sqrt(1.0/3)
        self.N_query=N_query
        self.N_p_values=N_p_values
        self.liner_nb_q1=BiasFreeLinearProjector(dim_in,self.c*N_head)
        self.liner_nb_k1=BiasFreeLinearProjector(dim_in,self.c*N_head)
        self.liner_nb_v1=BiasFreeLinearProjector(dim_in,self.c*N_head)

        self.liner_nb_q2=BiasFreeLinearProjector(dim_in,N_head*N_query*3)
        self.liner_nb_k2=BiasFreeLinearProjector(dim_in,N_head*N_query*3)

        self.liner_nb_v3=BiasFreeLinearProjector(dim_in,N_head*N_p_values*3)

        self.liner_nb_z=BiasFreeLinearProjector(dim_z,N_head)
        self.lastlinear1=LinearProjector(N_head*dim_z,dim_in)
        self.lastlinear2=LinearProjector(N_head*c,dim_in)
        self.lastlinear3=LinearProjector(N_head*N_p_values*3,dim_in)
        self.gama = nn.ParameterList([nn.Parameter(torch.zeros(N_head))])
        self.cos_f=nn.CosineSimilarity(dim=-1)

    def forward(self,s,z,rot,trans):
        L=s.shape[0]
        q1=self.liner_nb_q1(s).reshape(L,self.N_head,self.c) # Lq,
        k1=self.liner_nb_k1(s).reshape(L,self.N_head,self.c)
        v1=self.liner_nb_v1(s).reshape(L,self.N_head,self.c) # lv,h,c

        attmap=torch.einsum('ihc,jhc->ijh',q1,k1) * self.c_squ # Lq,Lk_v,h
        bias_z=self.liner_nb_z(z) # L L h

        q2 = self.liner_nb_q2(s).reshape(L,self.N_head,self.N_query,3)
        k2 = self.liner_nb_k2(s).reshape(L,self.N_head,self.N_query,3)

        v3 = self.liner_nb_v3(s).reshape(L,self.N_head,self.N_p_values,3)

        q2 = transform_ipa_data(q2,rot,trans) # Lq,self.N_head,self.N_query,3
        k2 = transform_ipa_data(k2,rot,trans) # Lk,self.N_head,self.N_query,3

        dismap=((q2[:,None,:,:,:] - k2[None,:,:,:,:])**2).sum([3,4]) ## Lq,Lk, self.N_head,
        attmap = attmap + bias_z - F.softplus(self.gama[0])[None,None,:]*dismap*self.W_c*0.5
        o1 = (attmap[:,:,:,None] * z[:,:,None,:]).sum(1) # Lq, N_head, c_z
        o2 = torch.einsum('abc,dab->dbc',v1,attmap) # Lq, N_head, c
        o3 = transform_ipa_data(v3,rot,trans) # Lv, h, p* ,3
        o3 = inverse_transform_ipa_data( torch.einsum('vhpt,gvh->ghpt',o3,attmap),rot,trans) #Lv, h, p* ,3

        return self.lastlinear1(o1.reshape(L,-1)) + self.lastlinear2(o2.reshape(L,-1)) + self.lastlinear3(o3.reshape(L,-1))


class TorsionPredictor(nn.Module):
    """TorsionNet with significant implementation differences"""
    def __init__(
        self, 
        sequence_dim: int, 
        hidden_dim: int
    ):
        super().__init__()
        self.sequence_dim = sequence_dim
        self.hidden_dim = hidden_dim
        
        # Feature transformation blocks
        self.linear1 = LinearProjector(sequence_dim, hidden_dim)
        self.linear2 = LinearProjector(sequence_dim, hidden_dim)
        self.linear3 = LinearProjector(hidden_dim, hidden_dim)
        self.linear4 = LinearProjector(hidden_dim, hidden_dim)
        self.linear5 = LinearProjector(hidden_dim, hidden_dim)
        self.linear6 = LinearProjector(hidden_dim, hidden_dim)
        
        # Parameter prediction layers
        self.linear7_1 = LinearProjector(hidden_dim, 1)
        self.linear7_2 = LinearProjector(hidden_dim, 2)
        self.linear7_3 = LinearProjector(hidden_dim, 2)

    def process_features(
        self, 
        init_features: torch.Tensor,
        dynamic_features: torch.Tensor
    ) -> torch.Tensor:
        """Transform input features"""
        combined = F.relu(self.linear1(init_features) + self.linear2(dynamic_features))
        transformed = F.relu(self.linear3(combined))
        transformed = self.linear4(transformed) + combined
        transformed = F.relu(self.linear5(transformed))
        return self.linear6(transformed) + transformed

    def forward(
        self, 
        hidden_features: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Predict torsion parameters"""
        activated = F.relu(hidden_features)
        
        bond_length = self.linear7_1(activated)
        
        angle_vec = self.linear7_2(activated)
        angle_norm = torch.norm(angle_vec, dim=-1, keepdim=True)
        angle_parameters = angle_vec / (angle_norm + 1e-8)
        
        torsion_vec = self.linear7_3(activated)
        torsion_norm = torch.norm(torsion_vec, dim=-1, keepdim=True)
        torsion_parameters = torsion_vec / (torsion_norm + 1e-8)
        
        return bond_length, angle_parameters, angle_norm, torsion_parameters, torsion_norm

class TransitionModule(nn.Module):
    def __init__(self, c):
        super(TransitionModule, self).__init__()
        self.c = c
        self.norm1 = nn.LayerNorm(c)
        self.linear1 = LinearProjector(c, c)
        self.linear2 = LinearProjector(c, c)
        self.linear3 = LinearProjector(c, c)
        self.norm2 = nn.LayerNorm(c)

    def forward(self, s_):
        s = self.norm1(s_)
        s = F.relu(self.linear1(s))
        s = F.relu(self.linear2(s))
        s = s_ + self.linear3(s)
        return self.norm2(s)


class BackboneUpdate(nn.Module):
    def __init__(self, indim):
        super(BackboneUpdate, self).__init__()
        self.indim = indim
        self.linear = LinearProjector(indim, 6)
        torch.nn.init.zeros_(self.linear.linear.weight)
        torch.nn.init.zeros_(self.linear.linear.bias)

    def forward(self, s, L):
        pred = self.linear(s)
        rot = quat2rot(pred[..., :3], L)
        return rot, pred[..., 3:]  # rot, translation





class StructureModule(nn.Module):
    """StructureModule with significant implementation differences"""
    def __init__(
        self, 
        state_dim: int, 
        pair_dim: int, 
        layers: int , 
        hidden_dim: int
    ):
        super().__init__()
        self.s_dim = state_dim
        self.z_dim = pair_dim
        self.N_layer = layers
        self.c = hidden_dim
        
        # Feature normalization
        self.layernorm_s = nn.LayerNorm(state_dim)
        self.layernorm_z = nn.LayerNorm(pair_dim)
        
        # Core modules
        self.ipa = InvariantPointAttention(
            hidden_dim, pair_dim, hidden_dim
        )
        self.transition = TransitionModule(hidden_dim)
        self.bbupdate = BackboneUpdate(hidden_dim)
        
        # Initialize transformations
        self._initialize_transformation()
        self.torsionnet = TorsionPredictor(state_dim, hidden_dim)


            
    def _initialize_transformation(self) -> None:
        """Initialize identity transformation"""
        self.initial_rotation = torch.eye(3)
        self.initial_translation = torch.zeros(3)

    
    def compute_structural_parameters(
        self,
        sequence_features: torch.Tensor,
        pair_features: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor
    ) -> torch.Tensor:
        """Compute intermediate features"""
        ipa_output = self.ipa(sequence_features, pair_features, rotations, translations)
        features = sequence_features + ipa_output
        transition_output = self.transition(features)  # 添加残差连接
        return transition_output

    def apply_layer_transform(
        self,
        features: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rigid body transformation update"""
        new_rot, new_trans = self.bbupdate(features, length)
        return update_transformation_matrix(new_rot, new_trans,rotations, translations)

    def pred(
        self,
        features: torch.Tensor,
        pair_features: torch.Tensor,
        reference_points: torch.Tensor
    ) -> Dict[int, Tuple[torch.Tensor, ...]]:
        """Compute structure predictions over cycles"""
        sequence_len = features.size(0)
        rotations = self.initial_rotation.to(features.device).repeat(sequence_len, 1, 1)
        translations = self.initial_translation.to(features.device).repeat(sequence_len, 1)
        features = self.layernorm_s(features)
        pair_features = self.layernorm_z(pair_features)
        
        
        # Process through each layer
        for layer_idx in range(self.N_layer):
            features = self.compute_structural_parameters(features, pair_features, rotations, translations)
            rotations, translations = self.apply_layer_transform(features, rotations, translations, sequence_len)
        
        # Final processing
        features = self.compute_structural_parameters(features, pair_features, rotations, translations)
        rotations, translations = self.apply_layer_transform(features, rotations, translations, sequence_len)
        
        # Compute atom positions
        predicted_coords = reference_points + 0.0
        predicted_coords = transform_molecular_atoms(predicted_coords, rotations, translations)
        
        return predicted_coords, rotations, translations






class EvoBlock(nn.Module):
    def __init__(self,m_dim,z_dim,docheck=False):
        super(EvoBlock,self).__init__()
        self.msa_row=MSAFeaturesAttention(m_dim,z_dim)
        self.msa_col=SequenceFeaturesAttention(m_dim)
        self.msa_trans=MSATransition(m_dim)

        self.msa_opm=MSAOuterProductMean(m_dim,z_dim)

        self.pair_triout=PairwiseFeatureConstructor(z_dim)
        self.pair_triin =PairwiseFeatureInjector(z_dim)
        self.pair_tristart=PairwiseAttentionInitial(z_dim)
        self.pair_triend  =PairwiseAttentionTerminal(z_dim)
        self.pair_trans = PairTransition(z_dim)
        self.docheck=docheck


    def layerfunc_msa_row(self,m,z):
        return self.msa_row(m,z) + m
    def layerfunc_msa_col(self,m):
        return self.msa_col(m) + m
    def layerfunc_msa_trans(self,m):
        return self.msa_trans(m) + m
    def layerfunc_msa_opm(self,m,z):
        return self.msa_opm(m) + z

    def layerfunc_pair_triout(self,z):
        return self.pair_triout(z) + z
    def layerfunc_pair_triin(self,z):
        return self.pair_triin(z) + z
    def layerfunc_pair_tristart(self,z):
        return self.pair_tristart(z) + z
    def layerfunc_pair_triend(self,z):
        return self.pair_triend(z) + z
    def layerfunc_pair_trans(self,z):
        return self.pair_trans(z) + z
    def forward(self,m,z):

        m = m + self.msa_row(m,z)
        m = m + self.msa_col(m)
        m = m + self.msa_trans(m)
        z = z + self.msa_opm(m)
        z = z + self.pair_triout(z)
        z = z + self.pair_triin(z)
        z = z + self.pair_tristart(z)
        z = z + self.pair_triend(z)
        z = z + self.pair_trans(z)
        return m,z


class Evoformer(nn.Module):
    def __init__(self,m_dim,z_dim,docheck=False):
        super(Evoformer,self).__init__()
        self.layers=[16]
        self.docheck=docheck
        if docheck:
            pass
            #print('will do checkpoint')
        self.evos=nn.ModuleList([EvoBlock(m_dim,z_dim,True) for i in range(self.layers[0])])

    def layerfunc(self,layermodule,m,z):
        m_,z_=layermodule(m,z)
        return m_,z_


    def forward(self,m,z):
        for i in range(self.layers[0]):
            m,z=self.evos[i](m,z)

        return m,z

