import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class TransformerHeadless(nn.Module):
    def __init__(self, 
                 board_size=(15, 15),
                 d_model=256,
                 num_head=8,
                 num_layers=6,
                 num_chess_type=3,
                 dropout=0.1,
                 device='cuda'
    ):
        self.board_size = board_size
        self.d_model = d_model
        self.num_head = num_head
        self.num_layers = num_layers
        self.num_chess_type = num_chess_type
        self.dropout = dropout
        self.device = device

        self.chess_embedding = nn.Embedding(self.num_chess_type, d_model, device=self.device)
        self.pos_embedding = nn.Embedding(self.board_size[0] * self.board_size[1], device=self.device)
    
        self.encoder_layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=self.d_model, 
                nhead=self.num_head, 
                dropout=dropout,
                activation="gelu", 
                batch_first=True,
                device=self.device
                ) 
            for _ in range(self.num_layers)
            ])  # output: B * L * D
        
    def forward(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)
        assert x.size()[1:] == self.board_size  # torch.Size can directly compare with tuple

        batch_size = x.size(0)  # B
        seq_len = x.size(1) * x.size(2) # L

        x = x.reshape(batch_size, -1).long().to(self.device)    # B * L

        x = self.chess_embedding(x) # B * L * D
        x += self.pos_embedding(torch.arange(seq_len).reshape(1, -1))

        x = self.encoder_layers(x)  # B * L * D

        return x

class TransformerPlayer(nn.Module):
    def __init__(self, 
                 board_size=(15, 15),
                 d_model=256,
                 num_head=8,
                 num_layers=6,
                 num_chess_type=3,
                 dropout=0.1,
                 device='cuda'
    ):
        self.board_size = board_size
        self.d_model = d_model
        self.num_head = num_head
        self.num_layers = num_layers
        self.num_chess_type = num_chess_type
        self.dropout = dropout
        self.device = device
        
        self.transformer_headless = TransformerHeadless(
            board_size=self.board_size,
            d_model=self.d_model,
            num_head=self.num_head,
            num_layers=self.num_layers,
            num_chess_type=self.num_chess_type,
            device=self.device
        )
        
        self.output_layer = nn.Linear(d_model, 2, device=self.device),  # B * L * 2
        
    def forward(self, x):
        """
        returns log prob of each action: tensor of shape B * 2 * H * W
        """
        x = self.transformer_headless(x)    # B * L * D
        x = F.dropout(x, p=self.dropout)
        x = self.output_layer(x)    # B * L * 2
        x = F.log_softmax(x, dim=1)
        x.reshape(-1, *self.board_size, 2)
        x.permute(0, 3, 1, 2)   # B * 2 * H * W

        return x
    
    def decide(self, x, turn, sample=False):
        """
        returns action: list of (row, col)
        """
        batch_size = x.size(0)
        assert isinstance(turn, int) or (isinstance(turn, torch.tensor) and turn.size() == (batch_size,)) or (isinstance(turn, list) and len(turn) == batch_size)
        log_prob = self.forward(x)  # B * 2 * H * W
        log_prob = log_prob.reshape(batch_size, 2, -1)  # B * 2 * L
        log_prob = log_prob[torch.arange(log_prob.size(0)), turn]  # B * L
        if sample:
            action = torch.distributions.Categorical(logits=log_prob).sample()  # B
        else:
            action = torch.argmax(log_prob) # B

        ans = []
        for i in range(batch_size):
            ans.append((action[i] // self.board_size[1], action[i] % self.board_size[1]))
        return ans
    
class TransformerValueModel(nn.Module):
    def __init__(self, 
                 board_size=(15, 15),
                 d_model=256,
                 num_head=8,
                 num_layers=6,
                 num_chess_type=3,
                 dropout=0.1,
                 device='cuda'
    ):
        self.board_size = board_size
        self.d_model = d_model
        self.num_head = num_head
        self.num_layers = num_layers
        self.num_chess_type = num_chess_type
        self.dropout = dropout
        self.device = device
        
        self.transformer_headless = TransformerHeadless(
            board_size=self.board_size,
            d_model=self.d_model,
            num_head=self.num_head,
            num_layers=self.num_layers,
            num_chess_type=self.num_chess_type,
            device=self.device
        )
        
        self.output_layer_1 = nn.Linear(d_model, 1, device=self.device)
        self.output_layer_2 = nn.Linear(self.board_size[0] * self.board_size[1], 1, device=self.device)
        
    def forward(self, x):
        """
        returns value of the board: tensor of size B * 1
        """
        x = self.transformer_headless(x)    # B * L * D
        x = F.dropout(x, p=self.dropout)
        x = self.output_layer_1(x)    # B * L * 1
        x = x.squeeze(-1)   # B * L
        x = self.output_layer_2(x)    # B * 1

        return x
    
    @classmethod
    def from_player(cls, player):
        ans = cls(
            board_size=player.board_size,
            d_model=player.d_model,
            num_head=player.num_head,
            num_layers=player.num_layers,
            num_chess_type=player.num_chess_type,
            dropout=player.dropout,
            device=player.device)
        ans.transformer_headless.load_state_dict(player.transformer_headless.state_dict())
        return ans