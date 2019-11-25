"""Module containing the shared RNN model."""
import numpy as np
import collections

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import models.shared_base
import utils


logger = utils.get_logger()


def _get_dropped_weights(w_raw, dropout_p, is_training):
    """Drops out weights to implement DropConnect.

    Args:
        w_raw: Full, pre-dropout, weights to be dropped out.  self.w_hh_raw
        dropout_p: Proportion of weights to drop out.
        is_training: True iff _shared_ model is training.

    Returns:
        The dropped weights.

    TODO(brendan): Why does torch.nn.functional.dropout() return:
    1. `torch.autograd.Variable()` on the training loop
    2. `torch.nn.Parameter()` on the controller or eval loop, when
    training = False...

    Even though the call to `_setweights` in the Smerity repo's
    `weight_drop.py` does not have this behaviour, and `F.dropout` always
    returns `torch.autograd.Variable` there, even when `training=False`?

    The above TODO is the reason for the hacky check for `torch.nn.Parameter`.
    """
    dropped_w = F.dropout(w_raw, p=dropout_p, training=is_training)

    if isinstance(dropped_w, torch.nn.Parameter): #hacky check
        dropped_w = dropped_w.clone()

    return dropped_w


def isnan(tensor):
    return np.isnan(tensor.cpu().data.numpy()).sum() > 0


class EmbeddingDropout(torch.nn.Embedding):
    """Class for dropping out embeddings by zero'ing out parameters in the
    embedding matrix.

    This is equivalent to dropping out particular words, e.g., in the sentence
    'the quick brown fox jumps over the lazy dog', dropping out 'the' would
    lead to the sentence '### quick brown fox jumps over ### lazy dog' (in the
    embedding vector space).

    See 'A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks', (Gal and Ghahramani, 2016).
    """
    def __init__(self,
                 num_embeddings,   #corpus.num_tokens=1000
                 embedding_dim,    #args.shared_embed=1000隐藏维度
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 dropout=0.1,
                 scale=None):
        """Embedding constructor.

        Args:
            dropout: Dropout probability.
            scale: Used to scale parameters of embedding weight matrix that are
                not dropped out. Note that this is _in addition_ to the
                `1/(1 - dropout)` scaling.

        See `torch.nn.Embedding` for remaining arguments.
        """
        torch.nn.Embedding.__init__(self,
                                    num_embeddings=num_embeddings,
                                    embedding_dim=embedding_dim,
                                    max_norm=max_norm,
                                    norm_type=norm_type,
                                    scale_grad_by_freq=scale_grad_by_freq,
                                    sparse=sparse)
        self.dropout = dropout
        assert (dropout >= 0.0) and (dropout < 1.0), ('Dropout must be >= 0.0 '
                                                      'and < 1.0')
        self.scale = scale  #缩放参数

    def forward(self, inputs):  # pylint:disable=arguments-differ
        """Embeds `inputs` with the dropped out embedding weight matrix."""
        """
        training是nn.Module的属性，可以使用train(mode)来设置，作用如下：
        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.
        """
        if self.training:#self.training是nn.Module的属性，
            dropout = self.dropout
        else:
            dropout = 0

        if dropout:
            """
            ~Embedding.weight (Tensor) – the learnable weights of the module of shape (num_embeddings, embedding_dim) 
            initialized from N(0,1)
            """
            mask = self.weight.data.new(self.weight.size(0), 1)#self.weight是nn.Embedding的属性 self.weight[10000,1000] pytorch=0.3.1
            #mask = self.weight.detach().new_empty(self.weight.size(0), 1)  #mask[10000,1] pytorch=1.3.1
            """
            bernoulli_(p=0.5, *, generator=None) → Tensor
            Fills each location of self with an independent sample from \text{Bernoulli}(\texttt{p})Bernoulli(p) . self can have integral dtype.
            这里重置了mask，因此上面的创建方法使用new_empty或者其他的new_ones等等并不重要，这个mask是用于dropout的
            """
            mask.bernoulli_(1 - dropout)
            #刚开始mask是（10000，1）的，下一行再把他扩展成（10000,1000）
            mask = mask.expand_as(self.weight)  #Expand this tensor to the same size as other. self.expand_as(other) is equivalent to self.expand(other.size()).
            mask = mask / (1 - dropout)  #这里采用的是Inverted dropout方法，除1-dropout可以保证输出的激活值的期望不变
            masked_weight = self.weight * Variable(mask)
        else:
            masked_weight = self.weight
        if self.scale and self.scale != 1:
            masked_weight = masked_weight * self.scale
        #这里使用的是nn.functional和nn.Module没有太多本质区别，当没有可学习的参数的时候可以使用nn.functional否则
        #应该使用nn.Module
        """
        Input: LongTensor of arbitrary shape containing the indices to extract
        Weight: Embedding matrix of floating point type with shape (V, embedding_dim),
            where V = maximum index + 1 and embedding_dim = the embedding size
        Output: (*, embedding_dim), where * is the input shape
        """
        return F.embedding(inputs,
                           masked_weight,
                           max_norm=self.max_norm,
                           norm_type=self.norm_type,
                           scale_grad_by_freq=self.scale_grad_by_freq,
                           sparse=self.sparse)


class LockedDropout(nn.Module):
    # code from https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

def _clip_hidden_norms(hidden, hidden_norms, max_norm):
    """Clips the hiddens to `max_norm`.
    Args:
        hidden: A `torch.autograd.Variable` hidden state.
        hidden_norms: A `torch.FloatTensor` of hidden norms, size [batch_size].
        max_norm: Norm to clip to.
    Returns:
        The clipped hiddens.
    The caller should check first that `hidden` should be clipped, and this
    function should _only_ be called if at least one hidden needs to be
    clipped.
    """
    if utils.get_pytorch_version() < 0.4:
        # NOTE(brendan): This workaround for PyTorch v0.3.1 does everything in
        # numpy, because the PyTorch slicing and slice assignment is too flaky.
        hidden_norms = hidden_norms.cpu().numpy()
        clip_select = hidden_norms > max_norm
        clip_norms = hidden_norms[clip_select]

        mask = np.ones(hidden.size())
        normalizer = max_norm/clip_norms
        normalizer = normalizer[:, np.newaxis]

        mask[clip_select] = normalizer
        hidden *= torch.autograd.Variable(
            torch.FloatTensor(mask).cuda(), requires_grad=False)
    else:
        norm = hidden.data[hidden_norms > max_norm].norm(p=2, dim=-1)
        norm = norm.unsqueeze(-1)
        hidden[hidden_norms > max_norm] *= max_norm/norm

    return hidden

class RNN(models.shared_base.SharedModel):#继承关系RNN->models.shared_base.SharedModel->torch.nn.Module
    """Shared RNN model.这个shared_base在这里应该是没写完，只实现了一个计算参数数量的功能，其他和torch.nn.Module没有区别"""
    def __init__(self, args, corpus):
        """
        :param args: 命令行参数
        :param corpus: 数据集
        :properties
            decoder,从1000到10000的映射，一个全链接层
            encoder，一个自定义的EmbeddingDropout层，从10000，到1000的映射，可以设置dropout
            lockdrop，一个单独dropout层，作用？
            args.tie_weights：作用不明，用encode的权重覆盖decoder的权重？这也不是一个网络结构，怎么能覆盖呢？
            w_xc,w_xh,w_hc,w_hh,w_hc_raw，w_hh_raw，w_h，w_c，RNN的参数矩阵这里执行的就是一些初始化的工作
            static_init_hidden，作用，在forward中hidden不存在的时候可以设置一个hidden保证程序执行
        """
        models.shared_base.SharedModel.__init__(self)  #构造父类

        self.args = args
        self.corpus = corpus
        #linear实现从1000到10000的映射，也就是从隐藏维度映射回词的编号
        self.decoder = nn.Linear(args.shared_hid, corpus.num_tokens)  #shared_hid=1000,corpus.num_tokens=10000，在数据集中一共有10000个不同的词、
        #encoder实现从10000到1000的映射
        self.encoder = EmbeddingDropout(corpus.num_tokens,
                                        args.shared_embed,#shared_embed=1000隐藏维度
                                        dropout=args.shared_dropoute)  #shared_dropoute=0.1
        self.lockdrop = LockedDropout()#一个单独的dropout
        #???
        if self.args.tie_weights:
            self.decoder.weight = self.encoder.weight

        # NOTE(brendan): Since W^{x, c} and W^{h, c} are always summed, there
        # is no point duplicating their bias offset parameter. Likewise for
        # W^{x, h} and W^{h, h}.
        self.w_xc = nn.Linear(args.shared_embed, args.shared_hid)  #(1000,1000)
        self.w_xh = nn.Linear(args.shared_embed, args.shared_hid)

        # The raw weights are stored here because the hidden-to-hidden weights
        # are weight dropped on the forward pass.
        self.w_hc_raw = torch.nn.Parameter(
            torch.Tensor(args.shared_hid, args.shared_hid))
        self.w_hh_raw = torch.nn.Parameter(
            torch.Tensor(args.shared_hid, args.shared_hid))
        self.w_hc = None  #这两个参数是在forward中由w_hc_raw生成而来（dropout而来）
        self.w_hh = None

        self.w_h = collections.defaultdict(dict)  #collections.defaultdict(function_factory)一个函数工厂，里面的每个对象都是一个dict
        self.w_c = collections.defaultdict(dict)

        for idx in range(args.num_blocks):
            for jdx in range(idx + 1, args.num_blocks):
                #二维字典，形成一个下三角字典矩阵，存储的block的wh和wc
                self.w_h[idx][jdx] = nn.Linear(args.shared_hid,
                                               args.shared_hid,
                                               bias=False)
                self.w_c[idx][jdx] = nn.Linear(args.shared_hid,
                                               args.shared_hid,
                                               bias=False)
        #又把上面的字典矩阵转存到_w_h和_w_c中
        self._w_h = nn.ModuleList([self.w_h[idx][jdx] for idx in self.w_h for jdx in self.w_h[idx]])
        self._w_c = nn.ModuleList([self.w_c[idx][jdx] for idx in self.w_c for jdx in self.w_c[idx]])

        if args.mode == 'train':
            self.batch_norm = nn.BatchNorm1d(args.shared_hid)
        else:
            self.batch_norm = None
        #重置参数
        self.reset_parameters()
        #返回一个字典类keydefaultdict继承自defaultdic，自己实现了__missing__方法，当访问的key没有value的时候用init_hidden来初始化一个value，
        #这个value就是一个全零的Variable
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)#init_hidden是一个方法，返回一个全零的Variable

        logger.info('# of parameters: {0}'.format(format(self.num_parameters, ",d")))

    def forward(self,  # pylint:disable=arguments-differ
                inputs,  #[35,64]
                dag,   #有向无环图
                hidden=None,
                is_train=True):
        time_steps = inputs.size(0)  #temp_steps:35
        batch_size = inputs.size(1)  #batch_size:64

        is_train = is_train and self.args.mode in ['train']
        #真正调用F.dropout来dropout参数
        self.w_hh = _get_dropped_weights(self.w_hh_raw,
                                         self.args.shared_wdrop,
                                         self.training)
        self.w_hc = _get_dropped_weights(self.w_hc_raw,
                                         self.args.shared_wdrop,
                                         self.training)

        if hidden is None:
            hidden = self.static_init_hidden[batch_size]
        #input[35,64],一共是35*64个词，词汇表中一共是10000个词，encoder（Embedding）完成了对应的词向量的转换，即10000->1000的映射
        embed = self.encoder(inputs)

        if self.args.shared_dropouti > 0:#shared_dropouti:0.65,这是把embedding的结果给dropout了
            embed = self.lockdrop(embed, self.args.shared_dropouti if is_train else 0)

        # TODO(brendan): The norm of hidden states are clipped here because
        # otherwise ENAS is especially prone to exploding activations on the
        # forward pass. This could probably be fixed in a more elegant way, but
        # it might be exposing a weakness（什么弱点？） in the ENAS algorithm as currently
        # proposed.
        #这里采用了参数clip的方法来防止梯度爆炸
        # For more details, see
        # https://github.com/carpedm20/ENAS-pytorch/issues/6
        clipped_num = 0
        max_clipped_norm = 0
        h1tohT = []  #每个RNNcell产生的h
        logits = []  #每个RNNcell产生的结果
        for step in range(time_steps):
            x_t = embed[step]
            logit, hidden = self.cell(x_t, hidden, dag)

            hidden_norms = hidden.norm(dim=-1)
            max_norm = self.args.shared_max_hidden_norm
            if hidden_norms.data.max() > max_norm:
                hidden_norms = hidden_norms.data.cpu().numpy()

                clipped_num += 1
                if hidden_norms.max() > max_clipped_norm:
                    max_clipped_norm = hidden_norms.max()

                hidden = _clip_hidden_norms(hidden, hidden_norms, max_norm)

            logits.append(logit)
            h1tohT.append(hidden)

        if clipped_num > 0:
            logger.info('clipped {} hidden states in one forward pass.max clipped hidden state norm: {}'.format(clipped_num,max_clipped_norm))
        #torch.stack:Concatenates sequence of tensors along a new dimension.
        h1tohT = torch.stack(h1tohT)  #h1tohT(list,[35,[64,1000]]->Tensor(35,64,1000)
        output = torch.stack(logits)  #output(Tensor(35,64,1000))
        raw_output = output
        if self.args.shared_dropout > 0:
            output = self.lockdrop(output, self.args.shared_dropout if is_train else 0)

        dropped_output = output
        #decoded=decoder(1000->10000):output(35*64,1000)->(35*64,10000)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        #decoded(35*63,10000)->(35,64,10000)
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

        extra_out = {'dropped': dropped_output, 'hiddens': h1tohT, 'raw': raw_output}
        #decoded(35,64,10000),hidden(64,1000),extra_out{dropped_output(35,64,1000),h1tohT(35,64,1000),raw_output(35,64,1000)
        return decoded, hidden, extra_out

    #这个cell是用于计算得出的dag的前向计算的值的
    def cell(self, x, h_prev, dag):
        """Computes a single pass through the discovered RNN cell."""
        c = {}
        h = {}
        f = {}

        f[0] = self.get_f(dag[-1][0].name)
        c[0] = F.sigmoid(self.w_xc(x) + F.linear(h_prev, self.w_hc, None))
        h[0] = (c[0]*f[0](self.w_xh(x) + F.linear(h_prev, self.w_hh, None)) +
                (1 - c[0])*h_prev)

        leaf_node_ids = []
        q = collections.deque()
        q.append(0)

        # NOTE(brendan): Computes connections from the parent nodes `node_id`
        # to their child nodes `next_id` recursively, skipping leaf nodes. A
        # leaf node is a node whose id == `self.args.num_blocks`.
        #
        # Connections between parent i and child j should be computed as
        # h_j = c_j*f_{ij}{(W^h_{ij}*h_i)} + (1 - c_j)*h_i,
        # where c_j = \sigmoid{(W^c_{ij}*h_i)}
        #
        # See Training details from Section 3.1 of the paper.
        #
        # The following algorithm does a breadth-first (since `q.popleft()` is
        # used) search over the nodes and computes all the hidden states.
        while True:
            if len(q) == 0:
                break

            node_id = q.popleft()
            nodes = dag[node_id]

            for next_node in nodes:
                next_id = next_node.id
                if next_id == self.args.num_blocks:
                    leaf_node_ids.append(node_id)
                    assert len(nodes) == 1, 'parent of leaf node should have only one child'
                    continue

                w_h = self.w_h[node_id][next_id]
                w_c = self.w_c[node_id][next_id]

                f[next_id] = self.get_f(next_node.name)
                c[next_id] = F.sigmoid(w_c(h[node_id]))
                h[next_id] = (c[next_id]*f[next_id](w_h(h[node_id])) +
                              (1 - c[next_id])*h[node_id])

                q.append(next_id)

        # TODO(brendan): Instead of averaging loose ends, perhaps there should
        # be a set of separate unshared weights for each "loose" connection
        # between each node in a cell and the output.
        #
        # As it stands, all weights W^h_{ij} are doing double duty by
        # connecting both from i to j, as well as from i to the output.

        # average all the loose ends
        leaf_nodes = [h[node_id] for node_id in leaf_node_ids]
        output = torch.mean(torch.stack(leaf_nodes, 2), -1)

        # stabilizing the Updates of omega
        if self.batch_norm is not None:
            output = self.batch_norm(output)

        return output, h[self.args.num_blocks - 1]
    #功能：初始化variable，即全零的Tensor
    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.shared_hid)
        #下面这个方法没什么用，就是应对不同输入形式的，结果都是返回一个全零的Variable
        return utils.get_variable(zeros, self.args.cuda, requires_grad=False)

    def get_f(self, name):
        name = name.lower()
        if name == 'relu':
            f = F.relu
        elif name == 'tanh':
            f = F.tanh
        elif name == 'identity':
            f = lambda x: x
        elif name == 'sigmoid':
            f = F.sigmoid
        return f
    #类内部没有使用
    def get_num_cell_parameters(self, dag):
        num = 0

        num += models.shared_base.size(self.w_xc)
        num += models.shared_base.size(self.w_xh)

        q = collections.deque()
        q.append(0)

        while True:
            if len(q) == 0:
                break

            node_id = q.popleft()
            nodes = dag[node_id]

            for next_node in nodes:
                next_id = next_node.id
                if next_id == self.args.num_blocks:
                    assert len(nodes) == 1, 'parent of leaf node should have only one child'
                    continue

                w_h = self.w_h[node_id][next_id]
                w_c = self.w_c[node_id][next_id]

                num += models.shared_base.size(w_h)
                num += models.shared_base.size(w_c)

                q.append(next_id)

        logger.debug('# of cell parameters: {format(self.num_parameters, ",d")}')
        return num
    #重置参数
    def reset_parameters(self):
        init_range = 0.025 if self.args.mode == 'train' else 0.04
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
