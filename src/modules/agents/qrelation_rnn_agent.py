import torch.nn as nn
import torch.nn.functional as F
from .qrelation_rnn_gnn import adjacency_and_create_graph, GraphPyG, GCN

class QrelationRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(QrelationRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.graph_library = args.graph_library
        if self.graph_library == "dgl":
            self.graph = GraphPyG(args, args.rnn_hidden_dim, args.n_actions)
        elif self.graph_library == "pyG":
            self.graph = GCN(args.rnn_hidden_dim, args.rnn_hidden_dim, args.n_actions)

        self.n_agents = args.n_agents
        self.distance_index = []

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        g = adjacency_and_create_graph(inputs, self.args.n_agents, dis_idx, self.graph_library)

        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)
        o = self.fc2(h)

        # 4 + (n_agent-1)*(4+m) + (n_agent)*(4+m) + m

        q = self.graph(g, o)
        return q.view(b, a, -1), h.view(b, a, -1)