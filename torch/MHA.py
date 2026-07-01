import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=4096,num_heads=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size,hidden_size)
        self.k_linear = nn.Linear(hidden_size,hidden_size)
        self.v_linear = nn.Linear(hidden_size,hidden_size)
        self.o_linear = nn.Linear(hidden_size,hidden_size)
    
    def forward(self,hidden_state,causal_mask=None,past_key_value=None,use_cache=False):
        batch_size = hidden_state.size(0)
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        query = query.view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
        key = key.view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
        value = value.view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)

        if past_key_value is not None:
            past_key,past_value = past_key_value
            key = torch.cat([past_key,key],dim=2)
            value = torch.cat([past_value,value],dim=2)
        
        new_past_key_value = (key,value) if use_cache else None
        
        scale = self.head_dim** -0.5

        attention_score = torch.matmul(query,key.transpose(-1,-2))*scale

        if causal_mask is not None:
            attention_score = attention_score.masked_fill(causal_mask==0,float("-inf"))
            attention_soft = F.softmax(attention_score,dim=-1)
            output=torch.matmul(attention_soft,value)
            output=output.transpose(1,2).contiguous()
            output = output.view(batch_size,-1,self.num_heads*self.head_dim)
            output=self.o_linear(output)
            return (output,new_past_key_value) if use_cache else output