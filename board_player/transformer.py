import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

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
        
        # policy
        self.policy_head_linear = nn.Linear(d_model, 2, device=self.device)

        # value
        self.value_head_linear = nn.Linear(d_model, 2, device=self.device)

    def forward(self, x, turn=None):
        if isinstance(x, list):
            x = torch.tensor(x)
        if isinstance(turn, list):
            turn = torch.tensor(turn)
        assert x.size()[1:] == self.board_size  # torch.Size can directly compare with tuple
        if turn is not None:
            assert turn.size() == (x.size(0),)
        x = x.to(self.device)
        if turn is not None:
            turn = turn.to(self.device)

        batch_size = x.size(0)  # B
        seq_len = x.size(1) * x.size(2) # L

        x = x.reshape(batch_size, -1).long().to(self.device)    # B * L

        x = self.chess_embedding(x) # B * L * D
        x += self.pos_embedding(torch.arange(seq_len).reshape(1, -1))

        x = self.encoder_layers(x)  # B * L * D

        # policy
        policy = F.dropout(x, p=self.dropout)
        policy = self.policy_head_linear(x) # B * L * 2
        policy = policy.permute(0, 2, 1)  # B * 2 * L
        policy = F.log_softmax(policy, dim=2)   # B * 2 * L
        
        if turn is not None:
            policy = policy[torch.arange(batch_size), turn]  # B * L

        # value
        value = F.dropout(x, p=self.dropout)
        value = self.value_head_linear(x)   # B * L * 2
        value = torch.sigmoid(value)
        value = value.mean(dim=1)    # B * 2
        
        if turn is not None:
            value = value[torch.arange(batch_size), turn]  # B

        return policy, value
    
    def decide(self, x, turn, sample=False):
        """
        returns action: list of (row, col)
        """
        log_prob, _ = self.forward(x, turn) # B * L
        if sample:
            actions = torch.distributions.Categorical(logits=log_prob).sample()  # B
        else:
            actions = torch.argmax(log_prob) # B

        ans = []
        for action in actions:
            action = action.item()
            ans.append((action // self.board_size[1], action % self.board_size[1]))
        return ans