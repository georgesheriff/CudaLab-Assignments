import torch
import torch.nn as nn


# LSTM
class MyLSTMCellFromEquations(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input gate parameters
        self.W_ii = nn.Linear(input_dim, hidden_dim)
        self.W_hi = nn.Linear(hidden_dim, hidden_dim)

        # Forget gate parameters
        self.W_if = nn.Linear(input_dim, hidden_dim)
        self.W_hf = nn.Linear(hidden_dim, hidden_dim)

        # Candidate cell parameters
        self.W_ig = nn.Linear(input_dim, hidden_dim)
        self.W_hg = nn.Linear(hidden_dim, hidden_dim)

        # Output gate parameters
        self.W_io = nn.Linear(input_dim, hidden_dim)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t, state):
        h_prev, c_prev = state

        i_t = torch.sigmoid(
            self.W_ii(x_t) + self.W_hi(h_prev)
        )

        f_t = torch.sigmoid(
            self.W_if(x_t) + self.W_hf(h_prev)
        )

        g_t = torch.tanh(
            self.W_ig(x_t) + self.W_hg(h_prev)
        )

        o_t = torch.sigmoid(
            self.W_io(x_t) + self.W_ho(h_prev)
        )

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class MyLSTMFromEquations(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, batch_first=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            current_input_dim = input_dim if layer_idx == 0 else hidden_dim

            self.layers.append(
                MyLSTMCellFromEquations(
                    input_dim=current_input_dim,
                    hidden_dim=hidden_dim
                )
            )

    def forward(self, x, state=None):
        if not self.batch_first:
            x = x.transpose(0, 1)

        batch_size, seq_len, _ = x.shape
        device = x.device

        if state is None:
            h = [
                torch.zeros(batch_size, self.hidden_dim, device=device)
                for _ in range(self.num_layers)
            ]

            c = [
                torch.zeros(batch_size, self.hidden_dim, device=device)
                for _ in range(self.num_layers)
            ]
        else:
            h, c = state

        outputs = []

        for t in range(seq_len):
            layer_input = x[:, t, :]

            for layer_idx, lstm_cell in enumerate(self.layers):
                h[layer_idx], c[layer_idx] = lstm_cell(
                    layer_input,
                    (h[layer_idx], c[layer_idx])
                )

                layer_input = h[layer_idx]

            outputs.append(layer_input)

        output = torch.stack(outputs, dim=1)

        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, (h_n, c_n)

# Convolutional LSTM 
import torch
import torch.nn as nn


class MyConvLSTMCellFromEquations(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        padding = kernel_size // 2

        self.W_xi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.W_hi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

        self.W_xf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.W_hf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

        self.W_xg = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.W_hg = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

        self.W_xo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.W_ho = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x_t, state):
        h_prev, c_prev = state

        i_t = torch.sigmoid(
            self.W_xi(x_t) + self.W_hi(h_prev)
        )

        f_t = torch.sigmoid(
            self.W_xf(x_t) + self.W_hf(h_prev)
        )

        g_t = torch.tanh(
            self.W_xg(x_t) + self.W_hg(h_prev)
        )

        o_t = torch.sigmoid(
            self.W_xo(x_t) + self.W_ho(h_prev)
        )

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t



class MyConvLSTMFromEquations(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size=3,
        num_layers=1,
        batch_first=True
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            current_input_channels = (
                input_channels if layer_idx == 0 else hidden_channels
            )

            self.layers.append(
                MyConvLSTMCellFromEquations(
                    input_channels=current_input_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size
                )
            )

    def forward(self, x, state=None):
        if not self.batch_first:
            x = x.transpose(0, 1)

        batch_size, seq_len, _, height, width = x.shape
        device = x.device

        if state is None:
            h = [
                torch.zeros(
                    batch_size,
                    self.hidden_channels,
                    height,
                    width,
                    device=device
                )
                for _ in range(self.num_layers)
            ]

            c = [
                torch.zeros(
                    batch_size,
                    self.hidden_channels,
                    height,
                    width,
                    device=device
                )
                for _ in range(self.num_layers)
            ]
        else:
            h, c = state

        outputs = []

        for t in range(seq_len):
            layer_input = x[:, t]

            for layer_idx, conv_lstm_cell in enumerate(self.layers):
                h[layer_idx], c[layer_idx] = conv_lstm_cell(
                    layer_input,
                    (h[layer_idx], c[layer_idx])
                )

                layer_input = h[layer_idx]

            outputs.append(layer_input)

        output = torch.stack(outputs, dim=1)

        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, (h_n, c_n)