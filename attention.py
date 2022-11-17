import torch.nn as nn
import torch

from fc import FullyConnectedLayer


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        # TODO: DICE acitivation function
        # TODO: attention weight normalization
        self.local_att = LocalActivationUnit(hidden_size=[64, 16], bias=[True, True], embedding_dim=embedding_dim, batch_norm=False)

    
    def forward(self, query_ad, user_behavior, user_behavior_length):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # user behavior length: size -> batch_size * 1
        # output              : size -> batch_size * 1 * embedding_size
        
        attention_score = self.local_att(query_ad, user_behavior)
        # 其实已经得到了每个 user_bahav 向量的weight了，和user behavior  相乘即可， 注意 user behavior的长度，每个sample的长度不同，因此需要一个mask
        # 去屏蔽后面的 attention score
        # print('attention_score.size()=', attention_score.size())  # B  * time_seq_len * 1
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        # print('attention_score.size()=', attention_score.size())  # B * 1 * time_seq_len
        
        # define mask by length
        user_behavior_length = user_behavior_length.type(torch.LongTensor)
        mask = torch.arange(user_behavior.size(1))[None, :] < user_behavior_length[:, None]
        # print('mask.size=', mask.size())  # B * 1 * time_seq
        # 这个mask 就是遮盖用，若 user_behav_len[i] 长度为N 则 mask[i] 前N个为TRUE，后面为FALSE
        # print(mask)
        
        # mask
        if torch.cuda.is_available():
            output = torch.mul(attention_score, mask.type(torch.cuda.FloatTensor))  # batch_size *
        else:
            output = torch.mul(attention_score, mask.type(torch.FloatTensor))  # batch_size *
        # print('output.size=', output.size())  # B 1 time_len
        # multiply weight
        output = torch.matmul(output, user_behavior)

        return output
        

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_size=[80, 40], bias=[True, True], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_size=hidden_size,
                                       bias=bias,
                                       dropout_rate=0.2,
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)

        self.fc2 = FullyConnectedLayer(input_size=hidden_size[-1],
                                       hidden_size=[1],
                                       bias=[True],
                                       dropout_rate=0.2,
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)
        # TODO: fc_2 initialization

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        queries = torch.cat([query for _ in range(user_behavior_len)], dim=1)  # b * time_seq * emb
        # print('queries.size()=', queries.size())  # torch.Size([3, 20, 4])  即 query复制 time_seq份

        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior], dim=-1)
        # print('attention_input.size()=', attention_input.size()) # attention_input.size()= torch.Size([3, 20, 16])
        attention_output = self.fc1(attention_input)
        attention_output = self.fc2(attention_output)

        return attention_output

if __name__ == "__main__":
    a = AttentionSequencePoolingLayer(embedding_dim=4)
    
    import torch
    query_ad = torch.randn((3, 1, 4))
    user_behav = torch.randn((3, 20, 4))
    user_behav_len = torch.ones((3, 1))
    user_behav_len[2] = 0
    print(user_behav_len.dtype)
    print(user_behav_len)
    out = a(query_ad, user_behav, user_behav_len)
    print('out.size()=', out.size(), out.squeeze(1).size())
    print(out[2])