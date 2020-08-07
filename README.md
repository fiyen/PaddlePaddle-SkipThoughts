# PaddlePaddle-SkipThoughts
Skip-Thought Vectors

在之前的NLP经典系列中，我们已经回顾了很多种词嵌入的方法，包括skip-gram, CBOW, GloVe, fast text等。这些词嵌入的方法无一不是对单词的嵌入。有人会想，对于单词的嵌入只能表征单词特征的空间分布，并不能得到更加高级的语义信息，比如一段话的意思。这篇论文便更进一步地将预测词语改变为了预测句子。论文引入了RNN结构来对各语句进行学习和预测，从形式上来看，与后来的Encoder-Decoder结构已经有很多相似之处（虽然还不成熟，也没有引入注意力机制）。在后续的研究中，一些在现在大名鼎鼎的模型，都多少受到这个模型的启发（如BERT）。

<h1>基本原理</h1>
<p>在这篇论文之前，研究者对于如何表示词语之上的语义空间已经有了很多有意义的尝试。比如最简单的表示一个语句的方法就是将语句中所有单词的向量叠加，这样简单粗暴的处理方式在一些领域（如文本分类）中可以取得相当不错的效果。但是毕竟方法太简单，其处理方式也类似词袋模型那样，很多细节上的处理都被忽视了。比如这种处理方式并不能很好地考虑词语之间顺序带来的语义的差别。在当时，Encoder-Decoder结构已被应用于文本翻译中，其结构形式可以很好地感知语序带来的差别，RNN作为Encoder进行训练时，隐含层的变化可以反映出语句意思的变化，因此其隐含层可以被用来作为语句向量空间的参考。</p>
<p>如何表示用一个隐含状态表示一个句子呢？给定一个句子s_i=\left{w_1,w_2, w_3,...,w_n\right}，其中<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>w</mi><mi>j</mi></msub></mrow>w_j</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.716668em;vertical-align:-0.286108em;" class="strut"></span><span class="mord"><span style="margin-right:0.02691em;" class="mord mathnormal">w</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.311664em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:-0.02691em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span style="margin-right:0.05724em;" class="mord mathnormal mtight">j</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.286108em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>是句中的第<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>j</mi></mrow>j</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.85396em;vertical-align:-0.19444em;" class="strut"></span><span style="margin-right:0.05724em;" class="mord mathnormal">j</span></span></span></span>个单词，RNN结构读入前<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>j</mi></mrow>j</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.85396em;vertical-align:-0.19444em;" class="strut"></span><span style="margin-right:0.05724em;" class="mord mathnormal">j</span></span></span></span>个单词，会输出一个隐含状态<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>h</mi><mi>j</mi></msub></mrow>h_j</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.980548em;vertical-align:-0.286108em;" class="strut"></span><span class="mord"><span class="mord mathnormal">h</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.311664em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span style="margin-right:0.05724em;" class="mord mathnormal mtight">j</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.286108em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>。而RNN读入<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>w</mi><mi>n</mi></msub></mrow>w_n</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.58056em;vertical-align:-0.15em;" class="strut"></span><span class="mord"><span style="margin-right:0.02691em;" class="mord mathnormal">w</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.151392em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:-0.02691em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">n</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>后，会输出<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>h</mi><mi>n</mi></msub></mrow>h_n</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.84444em;vertical-align:-0.15em;" class="strut"></span><span class="mord"><span class="mord mathnormal">h</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.151392em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">n</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>，由于这个隐含状态包含了前<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow>n</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.43056em;vertical-align:0em;" class="strut"></span><span class="mord mathnormal">n</span></span></span></span>个单词的所有信息，因此可以用<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>h</mi><mi>n</mi></msub></mrow>h_n</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.84444em;vertical-align:-0.15em;" class="strut"></span><span class="mord"><span class="mord mathnormal">h</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.151392em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">n</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>来表示这个句子（当然，由于没有引入注意力机制，所有单词的影响都被视为是相等的，这也是这篇论文不足的地方之一）。</p>
<p>假设我们有三句话<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><msub><mi>s</mi><mrow><mi>i</mi><mo>−</mo><mn>1</mn></mrow></msub><mo separator="true">,</mo><msub><mi>s</mi><mi>i</mi></msub><mo separator="true">,</mo><msub><mi>s</mi><mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub><mo fence="true">)</mo></mrow>\left(s_{i-1}, s_{i}, s_{i+1}\right)</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:1em;vertical-align:-0.25em;" class="strut"></span><span class="minner"><span style="top:0em;" class="mopen delimcenter">(</span><span class="mord"><span class="mord mathnormal">s</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.311664em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span><span class="mbin mtight">−</span><span class="mord mtight">1</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.208331em;" class="vlist"><span></span></span></span></span></span></span><span class="mpunct">,</span><span style="margin-right:0.16666666666666666em;" class="mspace"></span><span class="mord"><span class="mord mathnormal">s</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.31166399999999994em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span><span class="mpunct">,</span><span style="margin-right:0.16666666666666666em;" class="mspace"></span><span class="mord"><span class="mord mathnormal">s</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.311664em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span><span class="mbin mtight">+</span><span class="mord mtight">1</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.208331em;" class="vlist"><span></span></span></span></span></span></span><span style="top:0em;" class="mclose delimcenter">)</span></span></span></span></span>，它们在文本序列中是相邻的，如下图所示：其中<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>s</mi><mi>i</mi></msub><mo>=</mo></mrow>s_{i}=</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.58056em;vertical-align:-0.15em;" class="strut"></span><span class="mord"><span class="mord mathnormal">s</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.31166399999999994em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span><span style="margin-right:0.2777777777777778em;" class="mspace"></span><span class="mrel">=</span></span></span></span> I could see the cat on the steps. <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>s</mi><mrow><mi>i</mi><mo>−</mo><mn>1</mn></mrow></msub><mo>=</mo></mrow>s_{i-1}=</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.638891em;vertical-align:-0.208331em;" class="strut"></span><span class="mord"><span class="mord mathnormal">s</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.311664em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span><span class="mbin mtight">−</span><span class="mord mtight">1</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.208331em;" class="vlist"><span></span></span></span></span></span></span><span style="margin-right:0.2777777777777778em;" class="mspace"></span><span class="mrel">=</span></span></span></span> I got back home. <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>s</mi><mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo></mrow>s_{i+1}=</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.638891em;vertical-align:-0.208331em;" class="strut"></span><span class="mord"><span class="mord mathnormal">s</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.311664em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span><span class="mbin mtight">+</span><span class="mord mtight">1</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.208331em;" class="vlist"><span></span></span></span></span></span></span><span style="margin-right:0.2777777777777778em;" class="mspace"></span><span class="mrel">=</span></span></span></span> This was strange.</p>
<p><img alt="" src="https://ai-studio-static-online.cdn.bcebos.com/81f4f5b95695465097884095ec655a73415aca7b323e479d9c8ac42060cefd59"></p>
<p>一个RNN结构用来接收<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>s</mi><mi>i</mi></msub></mrow>s_{i}</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.58056em;vertical-align:-0.15em;" class="strut"></span><span class="mord"><span class="mord mathnormal">s</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.31166399999999994em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>中的所有单词，最后输出一个隐含状态<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>h</mi><mi>i</mi></msup></mrow>h^i</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.824664em;vertical-align:0em;" class="strut"></span><span class="mord"><span class="mord mathnormal">h</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span style="height:0.824664em;" class="vlist"><span style="top:-3.063em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span></span></span></span></span></span></span></span>，这一层作为Encoder层。论文采用GRU结构完成这一步。同时，另一个RNN结构基于这个隐含状态<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>h</mi><mi>i</mi></msub></mrow>h_i</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.84444em;vertical-align:-0.15em;" class="strut"></span><span class="mord"><span class="mord mathnormal">h</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.31166399999999994em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>，预测出这个句子的前方和后方的句子都是什么，当然也是通过RNN持续输出若干个词来预测一个句子，在训练的时候会有一个终止符（句号'.'或图中所示的<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&lt;</mo><mi>e</mi><mi>o</mi><mi>s</mi><mo>&gt;</mo></mrow>&lt;eos&gt;</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.5782em;vertical-align:-0.0391em;" class="strut"></span><span class="mrel">&lt;</span><span style="margin-right:0.2777777777777778em;" class="mspace"></span></span><span class="base"><span style="height:0.5782em;vertical-align:-0.0391em;" class="strut"></span><span class="mord mathnormal">e</span><span class="mord mathnormal">o</span><span class="mord mathnormal">s</span><span style="margin-right:0.2777777777777778em;" class="mspace"></span><span class="mrel">&gt;</span></span></span></span>），如果输出了这个符号则表示句子终止。这一层是Decoder层，论文中有2个Decoder，分别用来预测句子前和句子后。</p>
<p>Encoder层是一个标准的GRU结构，但是由于Decoder需要用到Encoder传过来的<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>h</mi><mi>i</mi></msub></mrow>h_i</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.84444em;vertical-align:-0.15em;" class="strut"></span><span class="mord"><span class="mord mathnormal">h</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.31166399999999994em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>，便需要在标准的GRU结构上做一些改进，使GRU训练过程中能够引入<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>h</mi><mi>i</mi></msub></mrow>h_i</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.84444em;vertical-align:-0.15em;" class="strut"></span><span class="mord"><span class="mord mathnormal">h</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.31166399999999994em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>参与前向传播。这个结构该怎么改进呢？如下是文中给出的结构，我们姑且称之为Conditional GRU，引入了<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>C</mi><mi>z</mi></msub><mo separator="true">,</mo><msub><mi>C</mi><mi>r</mi></msub><mo separator="true">,</mo><mi>C</mi></mrow>C_z,C_r,C</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.8777699999999999em;vertical-align:-0.19444em;" class="strut"></span><span class="mord"><span style="margin-right:0.07153em;" class="mord mathnormal">C</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.151392em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:-0.07153em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span style="margin-right:0.04398em;" class="mord mathnormal mtight">z</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span><span class="mpunct">,</span><span style="margin-right:0.16666666666666666em;" class="mspace"></span><span class="mord"><span style="margin-right:0.07153em;" class="mord mathnormal">C</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.151392em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:-0.07153em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span style="margin-right:0.02778em;" class="mord mathnormal mtight">r</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span><span class="mpunct">,</span><span style="margin-right:0.16666666666666666em;" class="mspace"></span><span style="margin-right:0.07153em;" class="mord mathnormal">C</span></span></span></span>三个变量。</p>
<p><img alt="" src="https://ai-studio-static-online.cdn.bcebos.com/d710d6a32b3448ddbe389734537b7400a8f0b358545747738adf4dc63d2d755c"></p>
<p>Decoder层的隐含层与词向量的维度尺寸相等，通过每次输出的隐含层<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>h</mi><mi>t</mi></msub></mrow>h_t</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.84444em;vertical-align:-0.15em;" class="strut"></span><span class="mord"><span class="mord mathnormal">h</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.2805559999999999em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">t</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.15em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>，将其与所有单词的词向量进行点乘，而将要预测的词便从这些词中选择点乘结果最大的。文中将点乘结果作为指数，以<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>e</mi></mrow>e</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.43056em;vertical-align:0em;" class="strut"></span><span class="mord mathnormal">e</span></span></span></span>为底，某个词被预测到的概率与这个值呈正比。如下式：</p>
<p><img alt="" src="https://ai-studio-static-online.cdn.bcebos.com/b09254f77eef438f8c6190307e7d26143318a9cbd9404750b9cd7a70f928e78a"></p>
<p>模型的目标是最大化所有预测词被正确预测的概率，包括<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>s</mi><mrow><mi>i</mi><mo>−</mo><mn>1</mn></mrow></msub></mrow>s_{i-1}</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.638891em;vertical-align:-0.208331em;" class="strut"></span><span class="mord"><span class="mord mathnormal">s</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.311664em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span><span class="mbin mtight">−</span><span class="mord mtight">1</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.208331em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>和<span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>s</mi><mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub></mrow>s_{i+1}</math></span><span aria-hidden="true" class="katex-html"><span class="base"><span style="height:0.638891em;vertical-align:-0.208331em;" class="strut"></span><span class="mord"><span class="mord mathnormal">s</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span style="height:0.311664em;" class="vlist"><span style="top:-2.5500000000000003em;margin-left:0em;margin-right:0.05em;"><span style="height:2.7em;" class="pstrut"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span><span class="mbin mtight">+</span><span class="mord mtight">1</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span style="height:0.208331em;" class="vlist"><span></span></span></span></span></span></span></span></span></span>中的词。公式如下：</p>
<p><img alt="" src="https://ai-studio-static-online.cdn.bcebos.com/d517f6b9ef2b4b0db2fd46d97a9e3af5762eaab821cd4d3db65a5c28eaf68032"></p>

GRU 以及Conditional GRU
GRU
标准的GRU结构我已在项目NLP经典之七：理解LSTM和GRU - Paddle实现中进行了详细的介绍，感兴趣的可以参考一下。下边我们重点介绍一下Conditional GRU该如何实现。

Conditional GRU
Conditional GRU与GRU的结构基本相同，所不同的是，Conditional GRU需要接收一个新的参数。虽然Paddle没有提供Conditional GRU，但是我们通过改写Paddle自带的basic_gru来轻松实现。改写部分如下：

将GRUUnit中加入encode_hidden_size，其它参数与GRUUnit相同；

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
将C,Cr,CxC,C_r,C_xC,C 
r
​	
 ,C 
x
​	
 加入到GRUUnit中的权重中；

 self._gate_weight = self.create_parameter(
     attr=gate_param_attr,
     shape=[self._input_size + self._hiden_size + self._encode_hiden_size, 2 * self._hiden_size],
     dtype=self._dtype)

 self._candidate_weight = self.create_parameter(
     attr=candidate_param_attr,
     shape=[self._input_size + self._hiden_size + self._encode_hiden_size, self._hiden_size],
     dtype=self._dtype)
改写forward函数，将encode_hidden并到输入中，在使用中再拆分出来 pre_encode_hidden即为将pre_hidden以及encode_hidden合并后的变量；

 def forward(self, input, pre_encode_hidden):
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
将basic_gru中加入参数encode_hidden，以及encode_hidden_size，以便传入Encoder层的隐含层信息；

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
将gru所有隐含状态都输出，而不是只输出last_hidden。

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
我将改写后的代码放入了conditional_gru.py中以方便调用，感兴趣的人可以查看详情。

拓展词向量
在训练完成后，我们可能会遇到这样一个问题，那就是输入测试集后，由于已经训练的词向量中没有包含测试集中的某个词，造成查询失败而报错。本文作者给出了解决这个问题的一种方法，利用从更大的数据集得到的向量中进行线性映射，具体参考项目如何实现词向量扩充？试试词向量线性映射工具。其思路是先训练已有的词向量与更大数据集测到的词向量的线性映射关系，然后将训练未得到而大数据集中包含的词的向量线性映射给训练后的模型。

这里需要注意的一点是，由于训练和测试的词向量的数目不一样，测试中需要重新设置Embedding层，并且不与训练的Embedding共享参数。那么训练得到的Embedding参数怎么传给测试的Embedding层？我们可以先把训练得到的所有向量提取出来，然后训练的Embedding层利用传入预训练参数的方式传进去。如下：

def init_emb(self, for_test=False):
    """
    初始化Embedding层的参数
    :param for_test: 是否用于训练
    :return:
    """
    if not for_test:
        self.embedding = fluid.Embedding(size=[len(self.vocab) + 1, self.word_emb_dim], padding_idx=0,
                                         param_attr=fluid.ParamAttr(name='embedding',
                                                                    initializer=fluid.initializer.UniformInitializer(
                                                                        low=-0.1, high=0.1
                                                                    ),
                                                                    learning_rate=self.lr,
                                                                    trainable=True),
                                         dtype='float32')
    else:
        self.test_emb_pin += 1
        if len(self.extra_emb) > 0:
            extra_vecs = np.array(self.extra_emb)
            extend_vecs = np.concatenate((self.emb_numpy, extra_vecs), axis=1)
            extend_vecs = np.asarray(extend_vecs, dtype='float32')
            init = fluid.ParamAttr(name='test_embedding_' + str(self.test_emb_pin),
                                   initializer=fluid.initializer.NumpyArrayInitializer(extend_vecs),
                                   trainable=False)
            self.test_embedding = fluid.Embedding(size=[extend_vecs.shape[0], self.word_emb_dim],
                                                  padding_idx=0, param_attr=init, dtype='float32')
如果对预训练向量的传递方式感兴趣，可以参考项目Paddle实践之预训练向量工具在Paddle的应用。
