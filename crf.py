from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        # 从起始位置到各标签的转移分数
        self.start_transitions = nn.Parameter(torch.empty(num_tags)) # shape: (num_tags,)
        # 从各标签到结束位置的转移分数
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        
        # 从各标签到各标签的转移分数
        '''
        shape: (num_tags, num_tags)
        '''
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags)) # shape: (num_tags, num_tags)
        
        # 参数初始化
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum', # 指定对输出的处理方式
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
                
            reduction: Specifies  the reduction to apply to the output:
                【怎么处理每段路径上的概率分数？】
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
            
            给tags: 返回 log(P(y|x)),  如果要计算loss，还要取个负数
            
            不给tags: 返回 
        """
        self._validate(emissions, tags=tags, mask=mask) # 检查输入的维度和有效性
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        # 计算分子（路径得分）
        numerator = self._compute_score(emissions, tags, mask)
        
        # shape: (batch_size,)
        # 计算分母（归一化因子）
        denominator = self._compute_normalizer(emissions, mask)
        
        # shape: (batch_size,)
        # 计算对数似然
        llh = numerator - denominator
        
        # 对结果进行归约
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)
    
    

    
    
    
    
    
    
    
    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        
        '''
         检查 emission， mask， tags 的维度
        '''
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            
            if not no_empty_seq and not no_empty_seq_bf:
                # 每个batch对应的第一个token mask 必须是1
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        '''
         计算路径得分
        '''
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all() # 所有batch上的第一个token的mask必须为True

        seq_length, batch_size = tags.shape
        mask = mask.float()
        
        '''
           以下处理序列的第一个时间步
        '''

        # Start transition score and first emission
        # start_transitions.shape: (num_tags,)
        # tags[0]: (batch_size,) 取出每个序列 在第0个时间步 的tag下标
        # score.shape = (batch_size,)
        
        # 假设 tag[0] = [12, 30, 9, 2, 1]
        # start_transitions[[12, 30, 9, 2, 1]] = [0.2, 0.3, 0.1, 0.05, 0.0]
        # 就是通过一个tag下标列表，来获得一个tag转移概率列表，作为初始的转移分数
        score = self.start_transitions[tags[0]]
        
        '''
         score 这么赋值的含义：
            初始化每个序列的得分为从起始状态转移到其第一个标签的分数。
        '''
        
        # 拿到 每个序列 在第0个时间步 的发射分数（发射到tag[0]中的标签)
        score += emissions[0, torch.arange(batch_size), tags[0]]
        '''
            为每个序列的初始得分添加其在第一个时间步的发射分数。
            完成了路径得分的第一步计算，即初始转移分数加上第一个时间步的发射分数。
            
            
            emissions[0]
                形状：(batch_size, num_tags)
                含义：第一个时间步的发射分数，针对每个序列和每个可能的标签。
            
            emissions[0, torch.arange(batch_size), tags[0]]

                操作：对于每个序列 i，取出其在时间步 0、标签为 tags[0][i] 的发射分数。
                索引方式：emissions[时间步, 序列索引, 标签索引]
                形状：(batch_size,)
                含义：每个序列在第一个时间步实际标签的发射分数。
        '''

        for i in range(1, seq_length):
            '''
            转移得分：
                从前一个标签转移到当前标签的得分，乘以对应的 mask，确保只在有效位置计算。
            发射得分：
                当前标签的发射得分，乘以 mask。
            '''
            
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            # mask[i].shape: (batch_size,)
            # self.transitions.shape: (num_tags, num_tags)
            # self.transitions[tags[i - 1], tags[i]].shape = (batch_size,)
            
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            # (batch_size, batch_size) x (batch_size, ) = (batch_size, )
 
            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # score.shape: (batch_size,)
            # mask[i].shape: (batch_size,)
            # emissions[i, torch.arange(batch_size), tags[i]].shape = (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]
        
        '''
        结束转移得分：
            从序列最后一个标签转移到结束状态的得分。
        '''

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        '''
            计算 Z(x)
        '''
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all() # 如果mask[0]中的所有值都为真，则断言成立

        seq_length = emissions.size(0)

        # Start transition score（转移分数） and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        
        '''
        
        #开始转移分数和首次发射分数，大小为

            #（batch_size，num_tags），
            # 其中对于每个批次，第j列存储第一个时间步标记为j的分数
            emissions[0] 存储了每个样本在第一个时间步的 emission 分数 #(batch_size, num_tags)
        '''
        # shape: (batch_size, num_tags)
        # # emissions: (seq_length, batch_size, num_tags)
        # self.start_transitions.shape = (num_tags,)
        # 每个样本的开始转移分数都相同
        score = self.start_transitions + emissions[0] 

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
    
    
    def _beam_search_decode(self, emissions: torch.Tensor, 
                            mask: Optional[torch.ByteTensor] = None)-> List[List[int]]:
        pass





if __name__ == '__main__':
    model = CRF(num_tags=5, batch_first=True)
    
    
    



