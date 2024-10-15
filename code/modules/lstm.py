import torch
import torch.nn as nn
import torch.nn.functional as F

'''
LSTMCell类负责在每个时间步上执行LSTM操作，包括处理输入数据x和前一时间步的隐藏状态hx（如果提供）。这个类计算四个门（输入门、遗忘门、单元候选和输出门）的值，并根据这些门的值更新单元状态和隐藏状态。
'''
class LSTMCell(nn.Module):
    """
    LSTM单元的实现。
    输入参数:
    - input_size: 输入数据的特征维度
    - hidden_size: 隐藏层的维度
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 输入门、遗忘门、单元状态和输出门的权重矩阵和偏置项
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, x, hx=None):
        """
        前向传播。
        输入参数:
        - x: 输入数据
        - hx: 包含上一时间步的隐藏状态和单元状态的元组(h, c)
        """
        if hx is None:
            # 如果hx是None，初始化为全0
            hx = x.new_zeros(x.size(0), self.hidden_size, requires_grad=False)
            cx = x.new_zeros(x.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hx

        # 计算所有门的值
        gates = (F.linear(x, self.weight_ih, self.bias_ih) +
                 F.linear(hx, self.weight_hh, self.bias_hh))

        # 将gates分为四部分，分别对应输入门、遗忘门、单元候选和输出门
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # 应用sigmoid函数到输入门、遗忘门和输出门；应用tanh函数到单元候选
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        # 更新单元状态
        cy = (forgetgate * cx) + (ingate * cellgate)
        # 计算当前的隐藏状态
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


'''
SimpleLSTM类使用LSTMCell处理一个完整的输入序列，每次处理序列中的一个元素，并通过一个全连接层产生最终的输出。
'''
class SimpleLSTM(nn.Module):
    """
    简单的LSTM网络，包含一个LSTM单元和一个全连接层。
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm_cell = LSTMCell(input_size, hidden_size)  # LSTM单元
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, x):
        outputs = []
        hn = None
        for i in range(x.size(1)):  # 按序列长度迭代
            hn = self.lstm_cell(x[:, i, :], hn)  # 更新隐藏状态和单元状态
            outputs.append(hn[0])
        out = outputs[-1]  # 取最后一个时间步的输出
        out = self.fc(out)  # 通过全连接层
        return out



