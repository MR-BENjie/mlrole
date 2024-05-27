import torch
from torch import nn
from typing import Any
import torch.nn.functional as F

class Module(torch.nn.Module):
    r"""
    Wraps ``torch.nn.Module`` to overload ``__call__`` instead of
    ``forward`` for better type checking.

    `PyTorch Github issue for clarification <https://github.com/pytorch/pytorch/issues/44605>`_
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        # To stop PyTorch from giving abstract methods warning
        pass

    def __init_subclass__(cls, **kwargs):
        if cls.__dict__.get('__call__', None) is None:
            return

        setattr(cls, 'forward', cls.__dict__['__call__'])
        delattr(cls, '__call__')

    @property
    def device(self):
        params = self.parameters()
        try:
            sample_param = next(params)
            return sample_param.device
        except StopIteration:
            raise RuntimeError(f"Unable to determine"
                               f" device of {self.__class__.__name__}") from None

class role_Encode(nn.Module):
    def __init__(self, in_features: int,
                 mlp_layer: int = 2,
                 mlp_hiddens=None,
                 mlp_out_features: int = 64,
                 hiddens=64,
                 hidden_layer: int = 2):
        super().__init__()

        if mlp_hiddens is None:
            mlp_hiddens = [64, 128]

        self.mlp_layer = mlp_layer
        inmlp_features = in_features
        self.inmlp = nn.ModuleList()
        for _, hidden in zip(range(mlp_layer), mlp_hiddens):
            self.inmlp.append(nn.Linear(inmlp_features, hidden))
            inmlp_features = hidden
        self.inmlp.append(nn.Linear(inmlp_features, mlp_out_features))

        self.lstm = nn.LSTM(input_size = mlp_out_features, hidden_size= hiddens, num_layers=hidden_layer)
        self.lstm_hidden = None
        self.lstm_c = None

        self.mlp_out_features = mlp_out_features


    def forward(self, input):
        input = torch.unsqueeze(input, dim=0)
        z = input
        for index in range(self.mlp_layer):
            z = self.inmlp[index](z)
            z = torch.relu(z)
        z = self.inmlp[self.mlp_layer](z)


        if self.lstm_hidden is None :
            encode, (self.lstm_hidden, self.lstm_c) = self.lstm(z)
        else:
            encode, (self.lstm_hidden, self.lstm_c) = self.lstm(z , (self.lstm_hidden, self.lstm_c))
        encode = torch.squeeze(encode,dim=0)
        #if self.GRU_hidden is None:
        #    self.GRU_hidden = torch.zeros((self., self.gru_hiddens)).to(z.device)
        #gru_output, self.GRU_hidden = self.GRU(z, self.GRU_hidden)
        #z = self.get_position_encoding(z)
        #encode = self.transform(z)
        return encode

    def get_position_encoding(self, x):
        max_length = x.size()[0]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        self.inv_timescales = self.inv_timescales.to(x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.mlp_out_features % 2))
        signal = signal.view(max_length, self.mlp_out_features)
        return signal

class MlroleNode(Module):
    def __init__(self,
                 role_types: int,
                 gru_hiddens: int=64,
                 mlp_layer: int = 2,
                 mlp_hiddens= None,

                 n_heads: int = 4,
                 is_concat: bool = False,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        super().__init__()

        if mlp_hiddens is None:
            mlp_hiddens = [64, 128]
        self.mlp_layer = mlp_layer

        # explicit relation layer
        self.role_types = role_types


        self.type_trans_mlp = nn.ModuleList()
        self.type_self_mlp = nn.Linear(gru_hiddens, gru_hiddens)
        self.type_mersion_mlp = nn.Linear(gru_hiddens * 2, gru_hiddens)

        for _ in range(role_types):
            self.type_trans_mlp.append(nn.Linear(gru_hiddens, gru_hiddens))
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

        # ambiguous relation layer
        self.gat = GraphAttentionV2Layer(in_features=gru_hiddens, out_features=gru_hiddens, n_heads= n_heads,
                 is_concat = is_concat,
                 dropout =dropout,
                 leaky_relu_negative_slope = leaky_relu_negative_slope,
                 share_weights = share_weights)

        inmlp_features = gru_hiddens*2
        self.type_define_mlp = nn.ModuleList()
        for _, hidden in zip(range(mlp_layer), mlp_hiddens):
            self.type_define_mlp.append(nn.Linear(inmlp_features, hidden))
            inmlp_features = hidden
        self.type_define_mlp.append(nn.Linear(inmlp_features, role_types))

    # input["hidden"] the hidden states (output of the Encode) of the agent i
    # input["ambiguous"](torch.tensor((batch_size, node, hidden_size))) the hidden states of the agent with ambiguous relation to the agents of i
    # input["types"](dict(type_number:list())) explicit relation match the hidden states (output of the Encode) of the adjacent agents with that explicit relation

    def forward(self, input):

        # explicit relation layer
        hidden = input["hidden"]
        H_ambiguous = input["ambiguous"]
        H_types = input["types"]
        #action = input["action"]
        hidden_1 = self.type_self_mlp(hidden)

        for type in H_types.keys():

            tmp = torch.zeros_like(hidden_1)
            hidden_agents = H_types[type]
            for agent in hidden_agents:
                tmp += self.type_trans_mlp[type](agent)
            tmp /= len(hidden_agents)

            hidden_1 = self.type_mersion_mlp(torch.cat((hidden_1, tmp),dim=-1))
            hidden_1 = self.activation(hidden_1)

        # ambiguous relation layer
        hidden_1 = torch.vstack((hidden_1, H_ambiguous))

        adjacent_mat = torch.ones((hidden_1.shape[0],hidden_1.shape[0],1),dtype=torch.bool).to(hidden_1.device)
        hidden_2 = self.gat(hidden_1, adjacent_mat)[0,:]
        hidden_2 = torch.unsqueeze(hidden_2, dim=0)

        # define the relation of the ambiguous agents
        type_define_input = hidden_2.repeat(H_ambiguous.shape[0], 1)


        type_define_input = torch.hstack([H_ambiguous, type_define_input])
        for index in range(self.mlp_layer):
            type_define_input = self.type_define_mlp[index](type_define_input)
            type_define_input = self.activation(type_define_input)

        type_define_output = self.type_define_mlp[self.mlp_layer](type_define_input)

        return torch.sigmoid(type_define_output)


class GraphAttentionV2Layer(Module):
    """
    ## Graph attention v2 layer
    This is a single graph attention v2 layer.
    A GATv2 is made up of multiple such layers.
    It takes
    $$\mathbf{h} = \{ \overrightarrow{h_1}, \overrightarrow{h_2}, \dots, \overrightarrow{h_N} \}$$,
    where $\overrightarrow{h_i} \in \mathbb{R}^F$ as input
    and outputs
    $$\mathbf{h'} = \{ \overrightarrow{h'_1}, \overrightarrow{h'_2}, \dots, \overrightarrow{h'_N} \}$$,
    where $\overrightarrow{h'_i} \in \mathbb{R}^{F'}$.
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        * `share_weights` if set to `True`, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        # The initial transformations,
        # $$\overrightarrow{{g_l}^k_i} = \mathbf{W_l}^k \overrightarrow{h_i}$$
        # $$\overrightarrow{{g_r}^k_i} = \mathbf{W_r}^k \overrightarrow{h_i}$$
        # for each head.
        # We do two linear transformations and then split it up for each head.
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)

        # #### Calculate attention score
        #
        # We calculate these for each head $k$. *We have omitted $\cdot^k$ for simplicity*.
        #
        # $$e_{ij} = a(\mathbf{W_l} \overrightarrow{h_i}, \mathbf{W_r} \overrightarrow{h_j}) =
        # a(\overrightarrow{{g_l}_i}, \overrightarrow{{g_r}_j})$$
        #
        # $e_{ij}$ is the attention score (importance) from node $j$ to node $i$.
        # We calculate this for each head.
        #
        # $a$ is the attention mechanism, that calculates the attention score.
        # The paper sums
        # $\overrightarrow{{g_l}_i}$, $\overrightarrow{{g_r}_j}$
        # followed by a $\text{LeakyReLU}$
        # and does a linear transformation with a weight vector $\mathbf{a} \in \mathbb{R}^{F'}$
        #
        #
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big(
        # \Big[
        # \overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}
        # \Big] \Big)$$
        # Note: The paper desrcibes $e_{ij}$ as
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big( \mathbf{W}
        # \Big[
        # \overrightarrow{h_i} \Vert \overrightarrow{h_j}
        # \Big] \Big)$$
        # which is equivalent to the definition we use here.

        # First we calculate
        # $\Big[\overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j} \Big]$
        # for all pairs of $i, j$.
        #
        # `g_l_repeat` gets
        # $$\{\overrightarrow{{g_l}_1}, \overrightarrow{{g_l}_2}, \dots, \overrightarrow{{g_l}_N},
        # \overrightarrow{{g_l}_1}, \overrightarrow{{g_l}_2}, \dots, \overrightarrow{{g_l}_N}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        # `g_r_repeat_interleave` gets
        # $$\{\overrightarrow{{g_r}_1}, \overrightarrow{{g_r}_1}, \dots, \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_r}_2}, \overrightarrow{{g_r}_2}, \dots, \overrightarrow{{g_r}_2}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        # Now we add the two tensors to get
        # $$\{\overrightarrow{{g_l}_1} + \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_l}_1} + \overrightarrow{{g_r}_2},
        # \dots, \overrightarrow{{g_l}_1}  +\overrightarrow{{g_r}_N},
        # \overrightarrow{{g_l}_2} + \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_l}_2} + \overrightarrow{{g_r}_2},
        # \dots, \overrightarrow{{g_l}_2}  + \overrightarrow{{g_r}_N}, ...\}$$
        g_sum = g_l_repeat + g_r_repeat_interleave
        # Reshape so that `g_sum[i, j]` is $\overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}$
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        # Calculate
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big(
        # \Big[
        # \overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}
        # \Big] \Big)$$
        # `e` is of shape `[n_nodes, n_nodes, n_heads, 1]`
        e = self.attn(self.activation(g_sum))
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        # $e_{ij}$ is set to $- \infty$ if there is no edge from $i$ to $j$.
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        # We then normalize attention scores (or coefficients)
        # $$\alpha_{ij} = \text{softmax}_j(e_{ij}) =
        # \frac{\exp(e_{ij})}{\sum_{j' \in \mathcal{N}_i} \exp(e_{ij'})}$$
        #
        # where $\mathcal{N}_i$ is the set of nodes connected to $i$.
        #
        # We do this by setting unconnected $e_{ij}$ to $- \infty$ which
        # makes $\exp(e_{ij}) \sim 0$ for unconnected pairs.
        a = self.softmax(e)

        # Apply dropout regularization
        a = self.dropout(a)

        # Calculate final output for each head
        # $$\overrightarrow{h'^k_i} = \sum_{j \in \mathcal{N}_i} \alpha^k_{ij} \overrightarrow{{g_r}_{j,k}}$$
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        # Concatenate the heads
        if self.is_concat:
            # $$\overrightarrow{h'_i} = \Bigg\Vert_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            # $$\overrightarrow{h'_i} = \frac{1}{K} \sum_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.mean(dim=1)