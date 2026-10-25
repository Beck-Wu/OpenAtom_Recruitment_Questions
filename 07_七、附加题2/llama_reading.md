# 1. LLaMA 和原始 Transformer 的关系是什么？LLaMA 使用的是哪一类 Transformer 架构？它的训练目标是什么？为什么“预测下一个 token”可以成为生成文本、回答问题等能力的基础？
- Decoder-only 的 Transformer
- 训练目的：预测下一个 token
- 因为要准确预测下一个词，模型必须理解语法、事实和上下文逻辑  

# 2. LLaMA 相比你已经学习的 Transformer 做了哪些关键改动？请说明 RMSNorm、SwiGLU、RoPE 分别解决或改善了什么问题。无需推导公式，但需要说明它们在模型中的作用。
- RMSNorm：
- - 作用：原始 Transformer 在子层之后做归一化（Post-Norm），LLaMA 将其移到子层输入之前（Pre-Norm），并采用 RMSNorm 替代 LayerNorm。
- - 改善的问题：极大地提高了深层网络训练的稳定性，防止梯度消失，让模型能顺利训练更深的层。  

- SwiGLU：
- - 作用：替换了传统的 ReLU 非线性激活函数。
- - 改善的问题：在相同的计算成本下，提升了模型的性能。  

- RoPE：
- - 作用：移除了绝对位置编码，在每一层注入旋转位置信息。
- - 改善的问题：让模型更好地捕捉相对位置关系，对处理长文本更友好。  

# 3. 为什么 LLaMA 不只是“参数更多的 Transformer”？论文如何处理训练数据来源、数据配比、训练 token 数与模型参数量之间的关系？为什么足够多且高质量的数据对小一些的模型也很重要？
- 在给定的计算预算下，用更多数据训练的小模型，性能优于用较少数据训练的大模型。同时，小模型推理成本更低。
- 来源：CommonCrawl、C4、Github、Wikipedia、Books、ArXiv、Stack Exchange。

- 配比：CommonCrawl 占大头（67%），C4（15%），代码和书籍等优质数据占约 18%。

- Token 数量与参数规模的关系：LLaMA 7B 和 13B 训练了 1000 Billion token，33B 和 65B 训练了 1400 Billion token。这远超此前相同参数模型的训练数据量。

- 为什么高质量海量数据对小模型重要：论文指出，7B 模型的性能在 1T token 后仍在提升（见图 1 训练损失曲线），表 3、表 4 显示 LLaMA-13B 在大多数基准上击败了 175B 的 GPT-3。这证明小模型（如 13B）通过高质量数据训练，可以超越大模型。

# 4. 论文中的 zero-shot / few-shot 评测是怎样完成的？选择论文中的一个评测任务说明：模型如何把一个分类、问答或推理任务转化为文本生成任务？这个过程是否更新了模型参数？它和微调有什么本质区别？
- Zero-shot：提供任务的文本描述 + 一个测试样例，模型直接生成答案或对选项排序。

- Few-shot：提供 1~64 个任务示例 + 一个测试样例，模型根据这些上下文生成答案。
- 以闭卷问答（Natural Questions）为例（第 3.2 节，图 3）：

- - 如何转化为文本生成：给模型输入 Prompt，例如 "Answer these questions: \n Q: Who wrote the book the origin of species? \n A: "。模型接着生成 "Charles Darwin"。然后通过 Exact Match对比答案，统计准确率。

- - 是否更新参数：完全不更新。模型只是根据上下文做推理（Inference）。

- - 与微调的区别：Zero/Few-shot 属于 In-context Learning（上下文学习），不改动模型权重；而微调（Fine-tuning）需要计算梯度，实质性地修改模型参数。

# 5. 预训练模型为什么不能直接等同于聊天助手？结合论文中的 instruction fine-tuning 内容，说明 Base Model 与 Instruct Model 的区别。指令微调改变了模型的哪些部分，又没有改变哪些部分？
- ase Model 与 Instruct Model 的区别（第 4 节）：

- - Base Model（如 LLaMA-65B）：只学会了文字接龙，虽然知识渊博，但不知道人类的交互习惯。你问它问题，它可能会继续续写你的问题，而不是回答。

- - Instruct Model（如 LLaMA-I）：经过指令微调后，学会了听从指令并回答。  
- 指令微调改变了什么，没改变什么：

- - 改变了：模型顶层的权重分布。让它学会了格式对齐（如回答问题时使用问答格式）。

- - 没有改变：模型的底层网络架构、绝大部分知识储备。论文第 4 节表 10 显示，仅仅经过极少量指令微调，LLaMA-I (65B) 在 MMLU 上的得分就从 63.4% 跃升至 68.9%，超越了同量级的 OPT-IML。这证明了微调只是解锁了模型原本就有的知识，改变了调用方式。

![](07_七、附加题2/链图.png)