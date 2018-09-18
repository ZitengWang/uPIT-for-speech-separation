import torch as th

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class PITNet(th.nn.Module):
    def __init__(self,
                 num_bins,
                 rnn="lstm",
                 num_spks=1,
                 num_layers=1,
                 hidden_size=512,
                 hidden_size_adapt=50,
                 dropout=0.0,
                 non_linear="relu",
                 bidirectional=True):
        super(PITNet, self).__init__()
        if non_linear not in ["relu", "sigmoid", "tanh"]:
            raise ValueError(
                "Unsupported non-linear type:{}".format(non_linear))
        self.num_spks = num_spks
        rnn = rnn.upper()
        if rnn not in ['RNN', 'LSTM', 'GRU']:
            raise ValueError("Unsupported rnn type: {}".format(rnn))
        self.rnn = getattr(th.nn, rnn)(
            num_bins,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional)
        self.drops = th.nn.Dropout(p=dropout)
        # one BLSTM layer and two FF layers
        self.ff1 = th.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, hidden_size * 2)
        self.ff2 = th.nn.Linear(hidden_size * 2, hidden_size)
        self.linear = th.nn.ModuleList([
            th.nn.Linear(hidden_size, num_bins)
            for _ in range(self.num_spks)
        ])
        # auxilary network for speaker adaptation
        self.adapt1 = th.nn.Linear(num_bins, hidden_size_adapt)
        self.adapt2 = th.nn.Linear(hidden_size_adapt, hidden_size_adapt)
        self.adapt_sigmoid = th.nn.Linear(hidden_size_adapt, hidden_size * 2)

        self.non_linear = {
            "relu": th.nn.functional.relu,
            "sigmoid": th.nn.functional.sigmoid,
            "tanh": th.nn.functional.tanh
        }[non_linear]
        self.num_bins = num_bins

    def forward(self, x, utt, train=True):
        is_packed = isinstance(utt, PackedSequence)
        # extend dim when inference
        if not is_packed and utt.dim() != 3:
            utt = th.unsqueeze(utt, 0)
        # using unpacked sequence
        # utt: N x T x D
        if is_packed:
            utt, _ = pad_packed_sequence(utt, batch_first=True)
        utt = self.adapt1(utt)
        utt = self.adapt2(utt)
        utt = 2*self.adapt_sigmoid(utt)
        # averaging over time 
        utt_info = th.mean(utt, 1, True)

        is_packed = isinstance(x, PackedSequence)
        # extend dim when inference
        if not is_packed and x.dim() != 3:
            x = th.unsqueeze(x, 0)
        x, _ = self.rnn(x)
        # using unpacked sequence
        # x: N x T x D
        if is_packed:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.drops(x)
        x = self.ff1(x)
        # adapt this layer like LHUC
        x = th.nn.functional.relu(x * utt_info)
        x = self.ff2(x)
        x = th.nn.functional.relu(x)

        m = []
        for linear in self.linear:
            y = linear(x)
            y = self.non_linear(y)
            if not train:
                y = y.view(-1, self.num_bins)
            m.append(y)
        return m

    def disturb(self, std):
        for p in self.parameters():
            noise = th.zeros_like(p).normal_(0, std)
            p.data.add_(noise)