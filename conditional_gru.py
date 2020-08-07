"""
定义gru_conditional
"""
from paddle.fluid.contrib import BasicGRUUnit
from paddle.fluid.layers.control_flow import StaticRNN
import copy
from paddle.fluid import layers


class ConditionalGRUUnit(BasicGRUUnit):
    """
    定义一个新的GRU类型，多了参数Cu,Cr,C。GRU的新公式:
    .. math::
        u_t & = actGate(W_ux xu_{t} + W_uh h_{t-1} + C_u h_i + b_u)

        r_t & = actGate(W_rx xr_{t} + W_rh h_{t-1} + C_r h_i + b_r)

        m_t & = actNode(W_cx xm_t + W_ch dot(r_t, h_{t-1}) + C_u h_i + C h_i + b_m)

        h_t & = dot(u_t, h_{t-1}) + dot((1-u_t), m_t)
    其他定义与GRU相同
        Args:
            name_scope(string) : The name scope used to identify parameters and biases
            hidden_size (integer): The hidden size used in the Unit.
            param_attr(ParamAttr|None): The parameter attribute for the learnable
                weight matrix. Note:
                If it is set to None or one attribute of ParamAttr, gru_unit will
                create ParamAttr as param_attr. If the Initializer of the param_attr
                is not set, the parameter is initialized with Xavier. Default: None.
            bias_attr (ParamAttr|None): The parameter attribute for the bias
                of GRU unit.
                If it is set to None or one attribute of ParamAttr, gru_unit will
                create ParamAttr as bias_attr. If the Initializer of the bias_attr
                is not set, the bias is initialized zero. Default: None.
            gate_activation (function|None): The activation function for gates (actGate).
                                      Default: 'fluid.layers.sigmoid'
            activation (function|None): The activation function for cell (actNode).
                                 Default: 'fluid.layers.tanh'
            dtype(string): data type used in this unit
    """

    def __init__(self,
                 name_scope,
                 encode_hidden_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 dtype='float32'):
        super(ConditionalGRUUnit, self).__init__(name_scope=name_scope,
                                                 hidden_size=hidden_size,
                                                 param_attr=param_attr,
                                                 bias_attr=bias_attr,
                                                 gate_activation=gate_activation,
                                                 activation=activation,
                                                 dtype=dtype)
        self._encode_hiden_size = encode_hidden_size

    def _build_once(self, input, pre_hidden):
        self._input_size = input.shape[-1]
        assert (self._input_size > 0)

        if self._param_attr is not None and self._param_attr.name is not None:
            gate_param_attr = copy.deepcopy(self._param_attr)
            candidate_param_attr = copy.deepcopy(self._param_attr)
            gate_param_attr.name += "_gate"
            candidate_param_attr.name += "_candidate"
        else:
            gate_param_attr = self._param_attr
            candidate_param_attr = self._param_attr

        self._gate_weight = self.create_parameter(
            attr=gate_param_attr,
            shape=[self._input_size + self._hiden_size + self._encode_hiden_size, 2 * self._hiden_size],
            dtype=self._dtype)

        self._candidate_weight = self.create_parameter(
            attr=candidate_param_attr,
            shape=[self._input_size + self._hiden_size + self._encode_hiden_size, self._hiden_size],
            dtype=self._dtype)

        if self._bias_attr is not None and self._bias_attr.name is not None:
            gate_bias_attr = copy.deepcopy(self._bias_attr)
            candidate_bias_attr = copy.deepcopy(self._bias_attr)
            gate_bias_attr.name += "_gate"
            candidate_bias_attr.name += "_candidate"
        else:
            gate_bias_attr = self._bias_attr
            candidate_bias_attr = self._bias_attr

        self._gate_bias = self.create_parameter(
            attr=gate_bias_attr,
            shape=[2 * self._hiden_size],
            dtype=self._dtype,
            is_bias=True)
        self._candidate_bias = self.create_parameter(
            attr=candidate_bias_attr,
            shape=[self._hiden_size],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input, pre_encode_hidden):
        #print('Im here!')
        #print(input.shape)
        pre_hidden, encode_hidden = layers.split(pre_encode_hidden,
                                                 num_or_sections=[self._hiden_size, self._encode_hiden_size],
                                                 dim=1)
        concat_input_hidden = layers.concat([input, pre_hidden, encode_hidden], 1)

        gate_input = layers.matmul(x=concat_input_hidden, y=self._gate_weight)

        gate_input = layers.elementwise_add(gate_input, self._gate_bias)

        gate_input = self._gate_activation(gate_input)
        r, u = layers.split(gate_input, num_or_sections=2, dim=1)

        r_hidden = r * pre_hidden

        candidate = layers.matmul(
            layers.concat([input, r_hidden, encode_hidden], 1), self._candidate_weight)
        candidate = layers.elementwise_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_hidden = u * pre_hidden + (1 - u) * c

        return new_hidden


def conditional_gru(input,
                    encode_hidden,
                    init_hidden,
                    encode_hidden_size,
                    hidden_size,
                    num_layers=1,
                    sequence_length=None,
                    dropout_prob=0.0,
                    bidirectional=False,
                    batch_first=True,
                    param_attr=None,
                    bias_attr=None,
                    gate_activation=None,
                    activation=None,
                    dtype="float32",
                    name="conditional_gru"):
    """
        定义一个新的GRU类型，多了参数Cu,Cr,C。GRU的新公式:
        .. math::
            u_t & = actGate(W_ux xu_{t} + W_uh h_{t-1} + C_u h_i + b_u)

            r_t & = actGate(W_rx xr_{t} + W_rh h_{t-1} + C_r h_i + b_r)

            m_t & = actNode(W_cx xm_t + W_ch dot(r_t, h_{t-1}) + C_u h_i + C h_i + b_m)

            h_t & = dot(u_t, h_{t-1}) + dot((1-u_t), m_t)
        其他定义与GRU相同
    Args:
       input (Variable): GRU input tensor,
                      if batch_first = False, shape should be ( seq_len x batch_size x input_size )
                      if batch_first = True, shape should be ( batch_size x seq_len x hidden_size )
       encode_hidden: The hidden state from the encoder of the GRU. If bidirectional is True, the encode_hidden is assert
                      to contain two parts, former half part is for forward direction, and later half part is for backward
                      direction.
       encode_hidden_size: The size of encode_hidden. If bidirectional is True, the encode_hidden_size includes the
                      former half part and the later half part, i.e., the actual size of encode_hidden is
                      encode_hidden_size / 2
       init_hidden(Variable|None): The initial hidden state of the GRU
                      This is a tensor with shape ( num_layers x batch_size x hidden_size)
                      if is_bidirec = True, shape should be ( num_layers*2 x batch_size x hidden_size)
                      and can be reshaped to tensor with ( num_layers x 2 x batch_size x hidden_size) to use.
                      If it's None, it will be set to all 0.
       hidden_size (int): Hidden size of the GRU
       num_layers (int): The total number of layers of the GRU
       sequence_length (Variabe|None): A Tensor (shape [batch_size]) stores each real length of each instance,
                       This tensor will be convert to a mask to mask the padding ids
                       If it's None means NO padding ids
       dropout_prob(float|0.0): Dropout prob, dropout ONLY works after rnn output of each layers,
                            NOT between time steps
       bidirectional (bool|False): If it is bidirectional
       batch_first (bool|True): The shape format of the input and output tensors. If true,
           the shape format should be :attr:`[batch_size, seq_len, hidden_size]`. If false,
           the shape format should be :attr:`[seq_len, batch_size, hidden_size]`. By default
           this function accepts input and emits output in batch-major form to be consistent
           with most of data format, though a bit less efficient because of extra transposes.
       param_attr(ParamAttr|None): The parameter attribute for the learnable
           weight matrix. Note:
           If it is set to None or one attribute of ParamAttr, gru_unit will
           create ParamAttr as param_attr. If the Initializer of the param_attr
           is not set, the parameter is initialized with Xavier. Default: None.
       bias_attr (ParamAttr|None): The parameter attribute for the bias
           of GRU unit.
           If it is set to None or one attribute of ParamAttr, gru_unit will
           create ParamAttr as bias_attr. If the Initializer of the bias_attr
           is not set, the bias is initialized zero. Default: None.
       gate_activation (function|None): The activation function for gates (actGate).
                                 Default: 'fluid.layers.sigmoid'
       activation (function|None): The activation function for cell (actNode).
                            Default: 'fluid.layers.tanh'
       dtype(string): data type used in this unit
       name(string): name used to identify parameters and biases

       Returns:
        rnn_out(Tensor),last_hidden(Tensor)
            - rnn_out is result of GRU hidden, with shape (seq_len x batch_size x hidden_size) \
              if is_bidirec set to True, shape will be ( seq_len x batch_sze x hidden_size*2)
            - last_hidden is the hidden state of the last step of GRU \
              shape is ( num_layers x batch_size x hidden_size ) \
              if is_bidirec set to True, shape will be ( num_layers*2 x batch_size x hidden_size),
              can be reshaped to a tensor with shape( num_layers x 2 x batch_size x hidden_size)
            - all_hidden is all the hidden states of the input, including the last_hidden and medium hidden states. \
              shape is (num_layers x seq_len x batch_size x hidden_size). if is_bidirec set to True, shape will be
              (2 x num_layers x seq_len x batch_size x hidden_size)
    """
    if bidirectional:
        encode_hidden, bw_encode_hidden = layers.split(encode_hidden, num_or_sections=2, dim=-1)
        encode_hidden_size = int(encode_hidden_size / 2)

    fw_unit_list = []

    for i in range(num_layers):
        new_name = name + '_layers_' + str(i)
        if param_attr is not None and param_attr.name is not None:
            layer_param_attr = copy.deepcopy(param_attr)
            layer_param_attr.name += '_fw_w_' + str(i)
        else:
            layer_param_attr = param_attr
        if bias_attr is not None and bias_attr.name is not None:
            layer_bias_attr = copy.deepcopy(bias_attr)
            layer_bias_attr.name += '_fw_b_' + str(i)
        else:
            layer_bias_attr = bias_attr

        fw_unit_list.append(
            ConditionalGRUUnit(new_name, encode_hidden_size, hidden_size, layer_param_attr,
                               layer_bias_attr, gate_activation, activation, dtype)
        )

    if bidirectional:
        bw_unit_list = []

        for i in range(num_layers):
            new_name = name + '_reverse_layers_' + str(i)
            if param_attr is not None and param_attr.name is not None:
                layer_param_attr = copy.deepcopy(param_attr)
                layer_param_attr.name += '_bw_w_' + str(i)
            else:
                layer_param_attr = param_attr
            if bias_attr is not None and bias_attr.name is not None:
                layer_bias_attr = copy.deepcopy(bias_attr)
                layer_bias_attr.name += '_bw_b_' + str(i)
            else:
                layer_bias_attr = bias_attr
            bw_unit_list.append(
                ConditionalGRUUnit(new_name, encode_hidden_size, hidden_size, layer_param_attr,
                                   layer_bias_attr, gate_activation, activation, dtype)
            )

    if batch_first:
        input = layers.transpose(input, [1, 0, 2])

    mask = None
    if sequence_length:
        max_seq_len = layers.shape(input)[0]
        mask = layers.sequence_mask(
            sequence_length, maxlen=max_seq_len, dtype='float32'
        )
        mask = layers.transpose(mask, [1, 0])

    direc_num = 1
    if bidirectional:
        direc_num = 2
    if init_hidden:
        init_hidden = layers.reshape(
            init_hidden, shape=[num_layers, direc_num, -1, hidden_size]
        )

    def get_single_direction_output(rnn_input,
                                    encode_hidden,
                                    unit_list,
                                    mask=None,
                                    direc_index=0):
        rnn = StaticRNN()
        #print(rnn_input.shape)
        with rnn.step():
            step_input = rnn.step_input(rnn_input)

            if mask:
                step_mask = rnn.step_input(mask)

            for i in range(num_layers):
                if init_hidden:
                    pre_hidden = rnn.memory(init=init_hidden[i, direc_index])
                else:
                    pre_hidden = rnn.memory(batch_ref=rnn_input,
                                            shape=[-1, hidden_size],
                                            ref_batch_dim_idx=1)
                encode_h = encode_hidden[i]
                pre_encode_hidden = layers.concat([pre_hidden, encode_h], axis=1)
                new_hidden = unit_list[i](step_input, pre_encode_hidden)

                if mask:
                    new_hidden = layers.elementwise_mul(
                        new_hidden, step_mask, axis=0) - layers.elementwise_mul(
                        pre_hidden, (step_mask - 1), axis=0)
                rnn.update_memory(pre_hidden, new_hidden)

                rnn.step_output(new_hidden)

                step_input = new_hidden
                if dropout_prob is not None and dropout_prob > 0.0:
                    step_input = layers.dropout(step_input, dropout_prob=dropout_prob, )

            rnn.step_output(step_input)

        rnn_out = rnn()

        last_hidden_array = []
        all_hidden_array = []  # 增加这个来得到所有隐含状态
        rnn_output = rnn_out[-1]

        for i in range(num_layers):
            last_hidden = rnn_out[i]
            all_hidden_array.append(last_hidden)
            last_hidden = last_hidden[-1]
            last_hidden_array.append(last_hidden)

        all_hidden_array = layers.concat(all_hidden_array, axis=0)
        all_hidden_array = layers.reshape(all_hidden_array, shape=[num_layers, input.shape[0], -1, hidden_size])
        last_hidden_output = layers.concat(last_hidden_array, axis=0)
        last_hidden_output = layers.reshape(last_hidden_output, shape=[num_layers, -1, hidden_size])

        return rnn_output, last_hidden_output, all_hidden_array

    fw_rnn_out, fw_last_hidden, fw_all_hidden = get_single_direction_output(
        input, encode_hidden, fw_unit_list, mask, direc_index=0)

    if bidirectional:
        bw_input = layers.reverse(input, axis=[0])
        bw_mask = None
        if mask:
            bw_mask = layers.reverse(mask, axis=[0])
        bw_rnn_out, bw_last_hidden, bw_all_hidden = get_single_direction_output(
            bw_input, bw_encode_hidden, bw_unit_list, bw_mask, direc_index=1)

        bw_rnn_out = layers.reverse(bw_rnn_out, axis=[0])

        rnn_out = layers.concat([fw_rnn_out, bw_rnn_out], axis=2)
        last_hidden = layers.concat([fw_last_hidden, bw_last_hidden], axis=1)
        all_hidden = layers.concat([fw_all_hidden, bw_all_hidden], axis=0)

        last_hidden = layers.reshape(
            last_hidden, shape=[num_layers * direc_num, -1, hidden_size])

        if batch_first:
            rnn_out = layers.transpose(rnn_out, [1, 0, 2])
        return rnn_out, last_hidden, all_hidden
    else:

        rnn_out = fw_rnn_out
        last_hidden = fw_last_hidden
        all_hidden = fw_all_hidden

        if batch_first:
            rnn_out = layers.transpose(rnn_out, [1, 0, 2])

        return rnn_out, last_hidden, all_hidden


if __name__ == '__main__':
    input_size = 128
    hidden_size = 256
    encode_hidden_size = 384
    num_layers = 1
    dropout = 0.5
    bidirectional = True
    batch_first = True
    batch_size = 20

    input = layers.fill_constant(shape=[batch_size, 10, input_size], dtype='float32', value=1.0)
    pre_hidden = layers.fill_constant(shape=[num_layers, batch_size, hidden_size * 2], dtype='float32', value=1.0)
    encode_hidden = layers.fill_constant(shape=[num_layers, batch_size, encode_hidden_size * 2], dtype='float32', value=1.0)
    sequence_length = layers.fill_constant(shape=[batch_size], dtype='int32', value=10)
    rnn_out, last_hidden, _ = conditional_gru(input, encode_hidden, pre_hidden, encode_hidden_size * 2, hidden_size,
                                              num_layers=num_layers, sequence_length=sequence_length,
                                              dropout_prob=dropout, bidirectional=bidirectional, batch_first=batch_first)