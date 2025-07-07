[toc]



# SFT

## 数据预处理

- **输入数据**：训练数据通常包括 **prompt**（输入）和 **response**（目标输出），每个样本由用户提供的问题和预期的答案组成。
- **输入格式**：模型通常需要将这些 **prompt** 和 **response** 转化为 **token IDs**，以便可以输入到神经网络中进行训练。

例如，对于以下数据：

1. 获取prompt/response

   + **Prompt**: `"What is the capital of France?"`
   + **Response**: `"Paris"`

2. **编码输入与输出**：

   - 使用预训练的 **tokenizer** 对 `prompt` 和 `response` 进行编码，得到 **input_ids** 和 **labels**
   - **`input_ids`** 是模型的输入，通常由 **prompt** 和 **response** 的 token 组成
   - **`labels`** 是模型的目标输出，通常是 **response** 的 token ID

   ~~~python
   model_inputs = {
       "input_ids": [
           [101, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 102],
       ],
       "labels": [
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 1100, 1200, 1300, 1400, 102],
       ]
   }
   ~~~

3. **填充和对齐**：

   - 将 **input_ids** 和 **labels** 填充到指定的最大长度 `max_seq_length`，确保它们具有相同的长度。

   ~~~python
   model_inputs = {
       "input_ids": [
           [101, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       ],
       "labels": [
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 1100, 1200, 1300, 1400, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       ]
   }
   ~~~

## **训练过程**

- 在训练过程中，模型会进行多个 **epoch** 的迭代。每个 epoch 都包含多个 **batch**，每个 batch 包含若干个样本。
- 每次训练时，模型通过以下步骤更新参数：
  1. **前向传播**：将每个 batch 的 `input_ids` 输入到模型，得到 `logits`。
  2. **计算损失**：通过计算 **logits** 和 **labels** 之间的交叉熵损失，得到当前模型的误差。
  3. **反向传播**：通过反向传播计算梯度，更新模型的权重。
  4. **优化器更新**：使用优化器（如 AdamW）根据梯度更新模型的参数。
  5. **梯度裁剪**：为了防止梯度爆炸，通常会对梯度进行裁剪，限制最大梯度值。
  6. **学习率更新**：根据学习率调度器更新学习率，通常在训练过程中逐步降低学习率。

# DPO

RLHF pipline主要有下面三个步骤：

+ SFT
+ Reward Modelling Phase：在base模型上做微调，使得模型能够听懂人类的指令
+ RL Fine-Tuning Phase：使得模型不仅能够听懂人类指令，还能产出符合人类偏好的回答

实际rlhf-ppo的训练中，存在【显存占据大】、【超参多】、【模型训练不稳定】等一系列问题。DPO考虑能不能避开奖励模型的训练，直接一步到位训练对齐模型。比起传统基于强化学习PPO的方式，它改进了以下两点：

+ 不再训练奖励模型，直接使用人类标注的偏好数据，一步到位训练对齐模型。
+ 不再使用强化学习的方法，通过数学推理，将原始的偏好对齐优化目标步步简化，最后通过类似于sft的方式，用更简化的步骤训练出对齐模型

<figure>
  <img src="llm.assets/2025-04-04 16-53-01屏幕截图.png" alt="图片描述" />
  <figcaption style="text-align: center;">DPO与RLHF技术比较</figcaption>
</figure>

## 步骤一：从优化目标中直接求解最优对齐模型

RLHF中优化的目标函数如下，要找到能最大化这个优化目标的对齐模型 $\pi$
$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi} \left[ r(x, y) \right] - \beta D_{\text{KL}} \left[ \pi(y|x) \| \pi_{\text{ref}}(y|x) \right]
$$
下面对目标函数进行改进：
$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi} \left[ r(x, y) \right] - \beta D_{\text{KL}} \left[ \pi(y|x) \| \pi_{\text{ref}}(y|x) \right] \\
= \max_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right] \quad(1)\\
= \min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta} r(x, y) \right] \quad(2)\\
= \min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[\log \frac{\pi (y|x) }{\frac{1}{Z(x)} \pi_{\text{ref}}(y|x) * \exp \left( \frac{1}{\beta} r(x,y) \right)}-\log Z(x) \right] \quad(3)
$$
公式1推导：根据kl散度的定义
$$
D_{\text{KL}}[\pi(y|x) \parallel \pi_{\text{ref}}(y|x)] = \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi_{\text{ref}}(y|x)}{\pi(y|x)} \right]
$$
公式1~2推导：除以 β，并取反（因此max改成min）

公式2~3推导：
$$
\begin{aligned}
\log \frac{\pi}{\pi_{\text{ref}}} - \frac{1}{\beta} r &= \log \frac{\pi}{\pi_{\text{ref}}} - \log \exp \left( \frac{1}{\beta} r \right) \\
&= \log \frac{ \pi}{\pi_{\text{ref}} * \exp \left( \frac{1}{\beta} r \right)} \\
&= \log \frac{\frac{1}{Z(x)}\pi }{\frac{1}{Z(x)} \pi_{\text{ref}} * \exp \left( \frac{1}{\beta} r \right)} \\
&= \log \frac{\pi }{\frac{1}{Z(x)} \pi_{\text{ref}} * \exp \left( \frac{1}{\beta} r \right)}-\log Z(x)
\end{aligned}
$$
观察推导结果的最后一步，发现 $\pi$ 本身是一个概率分布，如果能让分母 ${\frac{1}{Z(x)} \pi_{\text{ref}} * \exp \left( \frac{1}{\beta} r \right)}$也能变成一个概率分布的形式，那么能**以优化KL散度的视角来看待**。

基于此这个想法，**将 Z(x)构造成一个partition function（配分函数）**，即：
$$
{Z(x)} =\sum_y \pi_{\text{ref}}(y|x) * \exp \left( \frac{1}{\beta} r(x,y) \right)
$$
把这个 Z(x)带入最后一步左侧项的分母中，则有:
$$
\frac{1}{Z(x)} \pi_{\text{ref}} * \exp \left( \frac{1}{\beta} r \right)=\frac{ \pi_{\text{ref}}(y|x) * \exp \left( \frac{1}{\beta} r(x,y) \right)}{\sum_y \pi_{\text{ref}}(y|x) * \exp \left( \frac{1}{\beta} r(x,y) \right)}
$$
其中，分子表示给定某个(x,y)下的奖励期望；分母则表示给定x，所有可能的y的奖励期望之和。这相当于是一个归一化的操作，使得这一项的取值在[0,1]之间，也就满足了我们前面说的构造一个概率分布的需求。同时，由 Z(x)的定义我们知道，它是关于x的函数，且它准备优化的模型 π没有关系。

鉴于分子 $π(y|x)$已经是个显式的分布表示了，干脆把分母也写成一个显示的分布表示，定义为 $π^∗(y|x)$:
$$
π^∗(y|x)= {\frac{1}{Z(x)} \pi_{\text{ref}}(y|x) * \exp \left( \frac{1}{\beta} r(x,y) \right)} \quad(4)
$$
将公式4带入3中得：
$$
\min_{\pi} \mathbb{E}_{x \sim D} \left[ \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)}{\pi^*(y|x)} \right] - \log Z(x) \right] = 
\min_{\pi} \mathbb{E}_{x \sim D} \left[ D_{\text{KL}} \left( \pi(y|x) \| \pi^*(y|x) \right) - \log Z(x) \right]  \quad(5)
$$
观察式5 ，前面我们说过 Z(x)和准备优化的模型 $\pi$ 没有关系，所以可以把它忽略掉。那么现在只用关心KL散度这一项。我们知道KL散度在两个分布完全相等时达到最小，由此我们**可以写出模型的显式解**：
$$
π(y|x)=π^∗(y|x)= {\frac{1}{Z(x)} \pi_{\text{ref}}(y|x) * \exp \left( \frac{1}{\beta} r(x,y) \right)} \quad(6)
$$
对式 6 再做一个简单的改写：因为**以上推导都是在假设我们有一个固定的奖励函数的基础上进行的**，所以可以加一个下标 
r 来强调这一点，则式 6 可进一步被改写成:
$$
π_r(y|x)= {\frac{1}{Z(x)} \pi_{\text{ref}}(y|x) * \exp \left( \frac{1}{\beta} r(x,y) \right)} \quad(7)
$$
在正常的对齐训练中，这个奖励函数  $r(x,y)$ 不是任意的，它是先用数据训练出来的最优奖励模型，然后在这个最优奖励模型的基础上，我们再通过训练去找到最优对齐模型 $\pi$ 。最优的奖励模型 r 基于它训练出的最优的对齐模型 $\pi$ 依然满足式7 的关系，分别设它们为 $r^∗(x,y),π^∗(y|x)$ ，则有：
$$
π^*(y|x)= {\frac{1}{Z(x)} \pi_{\text{ref}}(y|x) * \exp \left( \frac{1}{\beta} r^*(x,y) \right)} \quad(7)
$$

## 步骤二：跳过奖励模型的训练

虽然现在得到了对齐模型 π 的显式解 ，但是很难直接利用起这个显式解形式，原因如下：

+ $Z(x)$的值很难估计
+ 省略训练奖励模型这个步骤，一步到位来训练对齐模型

基于上述第2个原因，可以先**从模型 π 的显式解中推出奖励函数 r** 的形式：
$$
r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$
既然能用最优对齐模型  $π^∗$ 表示出最优奖励模型  $r^∗$，那么直接把 $π^∗$ 代入到奖励模型的训练优化目标中去，就可以实现明面上训练奖励模型，实际上却一步到位训练出了对齐模型。

现在，问题回到“奖励模型的训练上”。通常使用“偏好排序”这种数据标注方式来对奖励模型进行训练，一般有2种偏好排序方法：

+ Bradley–Terry Model	

  只生成2个回答，<prompt x, chosen y1, reject y2>，即对于一个prompt，我们只生成2个回答，让人工对这两个回答的偏好做排序，我们希望奖励模型对chosen回答的给分尽量高于对reject回答的给分。

+ Plackett-Luce Model

  生成K个（K > 2）回答，<prompt x, y1, ..., yK>，假设人工标注后的偏好排序组合为 τ（比如人工人为偏好从大到小为y2 > y3 > y1 >... > yK），那么我们希望奖励模型对 τ 这个排序的总得分要大于其余任何可能的偏好排序。

### Bradley–Terry Model：只生成2个回答

在该模型下，假设有一个成对数据 (y1,y2) 分别表示chosen和reject回答，根据奖励模型对两个回答打出的分数，“y1打败y2的概率”可以被表示成：
$$
P(y_1 \succ y_2 | x) = \frac{\exp[r(x, y_1)]}{\exp[r(x, y_1)] + \exp[r(x, y_2)]}
$$
对于整个标注数据集 $D={[x^i,y_w^i,y_l^i ]}_{i=1}^N$ ，chosen打败reject的期望概率尽量大（其中，w=chosen，l=reject），所以奖励函数的总体优化目标可以设计成：
$$
\begin{aligned}
L_R(r_\phi, D) &= -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log P(y_w \succ y_l | x)\right] \\
&= -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \frac{\exp[r(x, y_w)]}{\exp[r(x, y_w)] + \exp[r(x, y_l)]}\right] \\
&= -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \frac{1}{1 + \frac{\exp[r(x, y_l)]}{\exp[r(x, y_w)]}}\right] \\
&= -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\frac{1}{ \log \left( 1 + \exp[r(x, y_l) - r(x, y_w)] \right)} \right] \\
&= -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma(r(x, y_w) - r(x, y_l))\right]
\end{aligned}
$$
假设最优奖励模型为 $r^∗(x,y)$，将其带入上面的优化目标中，有:
$$
\begin{aligned}
L_R(r_\phi, D) 
&= -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma(r^*(x, y_w) - r^*(x, y_l))\right]
\end{aligned}
$$
根据前文推导，最优的奖励模型又可以用最优的对齐模型 $π^∗(y|x)$来显式表示，把这个显式表示带入上面的优化目标中，则有：
$$
L_R(r_\phi, D) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left[ \beta \log \frac{\pi^*(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi^*(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right] \right]
$$
到这里已经把训练奖励模型的目标函数转化成只和对齐模型 $π$相关了。也就是说，可以一步到位，绕开训练奖励模型的过程，直接用标注好的【成对】偏好数据，以类似于sft的过程直接训练对齐模型。对上述式子再稍加改动，设等待训练的对齐模型为 $πθ$ ，则有：
$$
L_R(r_\phi, D) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left[ \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right] \right]
$$

### Plackett-Luce Model 模型：生成K（K>2）个回答



假设 τ 为人工标注出的真值排序，则我们希望 τ 能够打败其余任何一种可能的偏好排序。将“最优排序 τ 打败其余任何一种排序”的概率表示成：
$$
P(\tau | y_1, \dots, y_K, x) = \prod_{k=1}^{K} \frac{\exp[r(x, y_{\tau(k)})]}{\sum_{j=k}^{K} \exp[r(x, y_{\tau(j)})]}
$$
这个公式从直观上理解的话：

- 对于真实值 $\tau$ 中的第一个回答 $\tau_1$，它是人工标注的最高的数值，我们当然希望它的得分在 $\tau_1 \sim \tau_K$ 中占大头
- 对于真实值 $\tau$ 中的第二个回答 $\tau_2$，我们当然希望它的得分在 $\tau_2 \sim \tau_K$ 中占大头
- 对于真实值 $\tau$ 中的第三个回答 $\tau_3$，我们当然希望它的得分在 $\tau_3 \sim \tau_K$ 中占大头
- 以此类推，则不难理解上面在PT模型下概率P的表达式。

最优奖励函数 $r^∗(x,y)$ 代入上面的 P 中，则有：
$$
P(\tau | y_1, \dots, y_K, x) = \prod_{k=1}^{K} \frac{\exp[r^*(x, y_{\tau(k)})]}{\sum_{j=k}^{K} \exp[r^*(x, y_{\tau(j)})]}
$$
再用 $π^∗$ 去表示 $r^*$，则（这里可以把 Z(x)省略掉）：
$$
p^*(\tau | y_1, \dots, y_K, x) = \prod_{k=1}^{K} \frac{\exp\left( \beta \log \frac{\pi^*(y_{\tau(k)} | x)}{\pi_{\text{ref}}(y_{\tau(k)} | x)} \right)}{\sum_{j=k}^{K} \exp \left( \beta \log \frac{\pi^*(y_{\tau(j)} | x)}{\pi_{\text{ref}}(y_{\tau(j)} | x)} \right)}
$$
对于整个数据集来说，我们希望最优序列打败其余任何一个可能序列的期望概率尽量大，则多回答下DPO的目标函数：
$$
L_{\text{DPO}}(\pi_\theta, \pi_{\text{ref}}) = - \mathbb{E}_{\tau, y_1, \dots, y_K, x \sim D} \left[ \log \prod_{k=1}^{K} \frac{\exp \left( \beta \log \frac{\pi_\theta(y_{\tau(k)} | x)}{\pi_{\text{ref}}(y_{\tau(k)} | x)} \right)}{\sum_{j=k}^{K} \exp \left( \beta \log \frac{\pi_\theta(y_{\tau(j)} | x)}{\pi_{\text{ref}}(y_{\tau(j)} | x)} \right)} \right]
$$


# ORPO

《ORPO: Monolithic Preference Optimization without Reference Model》提出了一种新的语言模型偏好对齐方法——ORPO（Odds Ratio Preference Optimization），旨在简化现有的偏好对齐过程，消除对参考模型的需求。

现有的偏好对齐方法通常需要额外的参考模型，并经历多个阶段，如强化学习与人类反馈（RLHF）或直接偏好优化（DPO）。这些方法计算复杂，资源消耗大。

ORPO在监督微调（SFT）的基础上，引入了一个简单的对数赔率比惩罚项，以区分偏好和非偏好生成风格。这样，模型在微调过程中能够直接学习偏好对齐，而无需额外的参考模型或复杂的对齐阶段。

<figure>
  <img src="llm.assets/2025-04-03 17-42-32屏幕截图.png" alt="图片描述" />
  <figcaption style="text-align: center;">模型对齐技术比较</figcaption>
</figure>

## 目标函数

**ORPO的目标函数** 包含两个部分：

1. **监督微调损失 (SFT loss)**
2. **相对比率损失 (Relative Ratio Loss, LOR)**：通过优化生成选定响应（`yw`）与被拒绝响应（`yl`）之间的 **赔率比** 来进一步提高生成响应的质量。赔率比是基于生成两个响应的可能性进行比较的：选定响应和被拒绝响应。该损失鼓励模型生成更符合偏好数据的响应。

odds 表示模型生成输出序列 y 的可能性与不生成输出序列 y 的可能性的比值，定义如下：
$$
\text{odds}_\theta(y | x) = \frac{P_\theta(y | x)}{1 - P_\theta(y | x)}
$$
**赔率比**（odds ratio）是衡量生成某个响应的相对可能性的指标。通过计算生成选定响应（`yw`）和拒绝响应（`yl`）的赔率比，ORPO使得模型更加倾向于生成符合偏好数据的响应。
$$
OR_\theta(y_w, y_l) = \frac{\text{odds}_\theta(y_w | x)}{\text{odds}_\theta(y_l | x)}
$$
**LOR损失**通过最大化选定响应（`yw`）和拒绝响应（`yl`）的赔率比来进行优化，公式如下，其中 σ 是 Sigmoid 函数。这个损失函数的目标是最大化选定响应的赔率比相对于拒绝响应的赔率比。
$$
L_{OR} = -\log \sigma \left( \log \text{odds}_\theta(y_w | x) - \log \text{odds}_\theta(y_l | x) \right)
$$
目标函数的完整格式如下：
$$
L_{ORPO} = \mathbb{E}_{(x, y_w, y_l)} \left[ L_{SFT} + \lambda \cdot L_{OR} \right]
$$

## 梯度计算和优化

在训练过程中，ORPO通过计算 **梯度** 来更新模型参数。梯度包括两个部分：
$$
\nabla_{\theta} L_{OR} = \delta(d) \cdot h(d)
$$

1. **δ(d)**：该项对比了选定响应和拒绝响应的赔率比差异。如果选定响应的赔率比显著高于拒绝响应，δ(d)将接近0，表示模型无需进一步更新。如果模型更倾向于生成拒绝响应，则δ(d)作为惩罚项加速参数更新。
   $$
   \delta(d) = 1 + [\frac{\text{odds}_{\theta}(y_w|x)}{\text{odds}_{\theta}(y_l|x)}] ^{-1}
   $$
   
2. **h(d)**：表示选定响应和拒绝响应的梯度对比。该项对选定响应和拒绝响应的生成概率的梯度进行了加权对比，低概率响应的梯度会被放大，从而加速模型朝着生成高概率响应的方向进行调整。
   $$
   h(d) = \nabla_{\theta} \log P_{\theta}(y_w|x) \cdot \frac{1}{1 - P_{\theta}(y_w|x)} - \nabla_{\theta} \log P_{\theta}(y_l|x) \cdot \frac{1}{1 - P_{\theta}(y_l|x)}
   $$

# RAHF

RAHF算法思路非常简单，生成K个response，选择reward最大的进行 SFT 训练，具体步骤 如下：

+ 数据收集：数据收集可以利用正在训练的生成模型作为生成器，也可以利用预训练模型（例如LLaMA、ChatGPT，甚至人类）和训练模型的混合模型作为生成器，有利于提升数据生成的多样性和质量。
+ 数据排序：从Reward Model筛选出最符合人类需求的样本。
+ 模型微调：使用 SFT 进行监督微调

个人理解：需要进行 K 次rollout，效率不高，而且只使用了reward最高的数据进行训练，数据使用效率也不高。

# RLHF

## 方法总结

最近几年的各种方法其实都是在修改优势函数 $A$ 的计算方法。

+ PPO 使用GAE进行计算 $A$
+ ReMa、RLOO、GRPO 通过生成多个样本，使用$r-b$来计算 $A$
+ Reinforce 通过使用 KL-penal 作为b，使用$r-b$来计算 $A$

<figure style="text-align: center;">
  <img src="llm.assets/2025-04-14 18-33-44屏幕截图-1744627052106.png" alt="RLOO Illustration" style="width: 90%;">
  <figcaption style="font-size: 14px; color: gray;">图中GRPO与deepseek原论文有区别，原论文中计算回报r没有加入KL penalty，而是在损失函数中加入KL loss</figcaption>
</figure>



## Reinforce++

**REINFORCE++的核心思想是将PPO中的各种优化技巧整合到经典的强化学习算法REINFORCE中，以提升其性能和稳定性。**这样REINFORCE++不需要 Critic 从而节省计算资源，又有加持了 PPO 相关的优化技巧实现高效训练。 **REINFORCE++的特点是 比 GRPO 稳定比PPO快。**

### reinforce 与 actor-critic 比较

reinforce中通过直接优化策略来最大化期望的累积奖励，其梯度下降公式如下：
$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{S \sim \eta, A \sim \pi(S, \theta)}
\left[ \nabla_{\theta} \ln \pi(A | S, \theta) q_{\pi}(S, A) \right]
$$
其中$q_{\pi}(S,A)$定义如下：
$$
q_{\pi}(S_t, A_t) = \mathbb{E} [ G_t \mid S_t = S, A_t = A ]
$$
使用Monte Carlo方法计算得到：
$$
q_{\pi}(S_t, A_t) \approx G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k}
$$
为了解决reinforce的高方差问题，actor-critic引入了值函数网络来降低方差，但是可能会引入偏差。AC算法的梯度下降公式如下：
$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{S \sim \eta, A \sim \pi(S, \theta)}
\left[ \nabla_{\theta} \ln \pi(A | S, \theta) q_{\pi}(S, A)-v_{\pi}(S) \right]
$$
定义优势函数:
$$
A_{t}=q_{\pi}(s_t, a_t)-v_{\pi}(s_t)
$$
优势函数$A_t$的计算可以使用TD算法/GAE算法，使用TD算法：
$$
A_{t}=\delta_{t}=r_t+\gamma v(s_{t+1})-v(s_t)
$$
使用GAE算法：
$$
A_t^{GAE(\gamma, \lambda)}=\sum_{l=0}^{\infty}(\gamma \lambda)^l \delta_{t+l}
$$

### reinforce ++  tricks

#### Token 级别的 KL penalty

$r(s_t,a_t)$是actor网络在状态$s_t$下产生token $a_t$的即时奖励。算法在每个 token 奖励中添加了 kl penalty，具体公式如下：
$$
r(s_t,a_t)=I(s_t=[\text{EOS}])r(x,y)-\beta \text{KL}(t)
$$

$$
\text{KL}(t)=log \frac{\pi(s_t,a_t)}{\pi_{ref}(s_t,a_t)}
$$

其中EOS表示最后一个token，$r(x,y)$是基于模型/规则的奖励。

#### 小批量更新

与PPO一样：

+ 将训练数据分成更小的批次，而不是使用整个数据集进行更新。
+ 允许每个小批次进行多个参数更新，从而加速收敛并减少内存消耗。
+ 引入随机性，帮助避免局部最优解，并提高模型的泛化能力。

#### 奖励归一化与截断

+ 奖励归一化：对奖励进行标准化
+ 奖励截断：防止奖励太大/大小引起梯度爆炸

#### 优势计算

由于reinforce++没有使用critic网络，使用下面公式来估计优势：
$$
A_t(s_t,a_t)=r(x,y)-\beta \sum_{i=t}^{T}KL(i)
$$
归一化：对一个batch计算出的优势值进行均值和方差计算，然后进行归一化。

#### PPO-clip

与PPO算法一样，限制新旧策略之间的比率变化：
$$
\arg\max_{\pi_\theta} J(\pi_\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} A(s_t, a_t) \right]
$$

#### 伪代码

<img src="llm.assets/2025-04-14 18-57-12屏幕截图.png" alt="2025-04-14 18-57-12屏幕截图" style="zoom: 67%;" />

## PPO 

### 传统 RL PPO

PPO 采用actor-critic网络架构，**为了降低采样成本，提升训练效率**，重复以下流程：

+ 假设某次更新完毕后，得到策略 $\pi_{\theta_{old}}$
+ 用 $\pi_{old}$ 和环境交互，得到一批回合数据
+ **将把这一批回合数据重复使用k次**：即先把这批数据喂给 $\pi_{\theta_{old}}$，更新得到 $\pi_{1}$；再把这批数据喂给 $\pi_{1}$，更新得到 $\pi_{2}$ ；以此类推，做k次更新后，得到 $\pi_{\theta}$。（需要注意的是 一个batch数据会分成多个minibatch来进行更新）
+ 在这k次更新后，重复上面的过程，直到达到设定的停止条件为止。

由于生成的数据的策略和当前策略不同，需要进行重要性采样。

**重要性采样前，策略的梯度是**：
$$
\nabla J(\pi_\theta) = \mathbb{E}_t \left[ A(s_t, a_t) \nabla \log \pi_\theta(a_t | s_t) \right]
$$
**重要性采样后，策略的梯度是：**
$$
\nabla J(\pi_\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} A(s_t, a_t) \nabla \log \pi_\theta(a_t | s_t) \right]
$$
**根据重要性采样构造了新的策略梯度，那么actor优化目标就可以从这个策略梯度中反推出来：**
$$
\arg\max_{\pi_\theta} J(\pi_\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} A(s_t, a_t) \right]
$$
上面公式还有两个问题（不详细解释）：

+ 如何求解优势函数$A(s_t,a_t)$			

  GAE方法（借助critic网络）

+ 如何解决 $\pi_{\theta}$ 和 $\pi_{\theta_{old}}$差异过大的问题

  PPO clip / PPO penalty

### LLM PPO

LLM PPO下面几个需要注意的地方：

#### Token 级别的 KL penalty

与 Reinforce++一样，将Token 级别的 KL penalty加入了及时回报中
$$
r(s_t,a_t)=I(s_t=[\text{EOS}])r(x,y)-\beta \text{KL}(t)
$$

$$
\text{KL}(t)=log \frac{\pi(s_t,a_t)}{\pi_{ref}(s_t,a_t)}
$$

#### 优势计算

采用 GAE 方法，具体步骤如下：

+ 从序列的最后一个有效时间步开始，逐步向前计算 **TD误差**
  $$
  \delta_t = r_t + \gamma v(s_{t+1}) - v(s_t)
  $$

+ 使用 **GAE** 的公式递归计算优势
  $$
  A_t = \delta_t + \gamma A_{t+1}
  $$

归一化：对一个batch计算出的优势值进行均值和方差计算，然后进行归一化。

#### PPO-clip

使用 PPO clip限制新旧策略更新的幅度

#### Critic-loss

传统 critic loss计算公式如下：
$$
\text{Critic-loss} = r_t + \gamma * v_{t+1} - v_t
$$
LLM 中 对 critic loss 进行了优化，如下：

实际收益 returns 优化

+ 原始实际收益：$ returns = r_t + \gamma * v_{t+1}$

+ 优化实际收益：$returns = A_t + v_{old}$ 	注意：$v_{old}$ 是minibatch中保存的值，不会改变

用旧模型去约束新模型，使用batch中的$v_{old}$设计一个变化范围：

~~~python
# self.cliprange_value是一个常量 
# old_values: 老critic的预测结果 v_old
# values：新critic的预测结果 v_t	随着ppo-epoch\minibatch不断更新
values_clipped = torch.clamp( 
	values, 
	old_values - self.cliprange_value, 
	old_values + self.cliprange_value, )
~~~

Critic-loss的计算:

~~~python
# critic模型的loss定义为（预估预期收益-实际预期收益）**2 
vf_loss1 = (values - returns)**2 
vf_loss2 = (values_clipped - returns)**2 
# 同样，最后也是把critic loss平均到每个token上
vf_loss = 0.5 * torch.sum( torch.max(vf_loss1, vf_loss2) * mask) / mask.sum() 
~~~

## GRPO

PPO 中的值函数通常是一个与策略模型大小相当的模型，这带来了显著的内存和计算负担。

GRPO舍弃了critic 网络，从当前策略中生成一组输出（可以理解为：对同一个问题的不同尝试），计算每个输出的相对优势作为优势 $A_i$。

#### PPO 与 GRPO 比较

下图详细展示了PPO 与 GRPO 的区别：

+ PPO需要更新policy model和value model；GRPO只需要更新policy model
+ PPO中KL作用在 reference model 和 reward model中，用于调整奖励 r ；GRPO中KL作用目标函数中，会进行反向传播

<img src="llm.assets/2025-03-12 20-15-46屏幕截图.png" alt="2025-03-12 20-15-46屏幕截图" style="zoom:50%;" />

#### 优势计算

回报 $r_i$ 与 PPO 不同，为原始回报，不需要添加 KL penalty。

1. 结果监督（Outcome Supervision）

   奖励仅在输出的末尾给出，并且**所有token的优势都等于该输出的标准化奖励**。这种方法适用于只关心整体输出质量的任务。
   $$
   r = {r_1,r_2,r_3,……,r_G}
   $$

   $$
   \hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
   $$

2. 过程监督（Process Supervision）

   奖励不仅在输出的末尾给出，而是给每个推理步骤一个奖励，进而计算每个token的优势。此方法适用于更复杂的任务。
   $$
   R = \left\{ \left\{ r_{1}^{index(1)}, \ldots, r_{1}^{index(K1)} \right\}, \ldots, \left\{ r_{G}^{index(1)}, \ldots, r_{G}^{index(KG)} \right\} \right\}
   $$
   **奖励标准化**：
   $$
   \tilde{r}_{i}^{index(j)} = \frac{r_{i}^{index(j)} - \text{mean}(R)}{\text{std}(R)}
   $$
   **过程监督** 中，优势的计算不再是简单的对整个输出的奖励进行标准化，而是针对每个token计算它的**优势**。每个token的优势是它所在步骤及之后步骤的 **标准化奖励的总和**。可以通过以下公式计算：
   $$
   \hat{A}_{i,t} = \sum_{j \geq t} \tilde{r}_{i}^{index(j)}
   $$

#### 目标函数

GRPO的目标函数如下 ：

<img src="llm.assets/2025-03-12 21-36-29屏幕截图.png" alt="2025-03-12 21-36-29屏幕截图" style="zoom: 60%;" />

其中添加了KL loss：

<img src="llm.assets/2025-03-12 21-37-53屏幕截图.png" alt="2025-03-12 21-37-53屏幕截图" style="zoom:60%;" />

#### KL 散度计算

PPO 和 reinforce++ 用的第一种 k1 来估计 KL 散度，GRPO用的 k3

1. k1 （native estimator）:  $\log \frac{\pi(s,a)}{\pi_{ref}(s,a)}$
   + 估计器是无偏的，但是方差大，因为它对一半的样本是负的（$\pi(s,a)<\pi_{ref}(s,a)$）
   + 当$\pi(s,a))$与$\pi_{ref}(s,a)$差异大时，对数值会产生很大的变化
2. k2（low variance estimator）: $\frac{1}{2} (\log \frac{\pi(s,a)}{\pi_{ref}(s,a)})^2$
   + 估计器是有偏的，但是方差低
   + 实证偏差很小，因为保证了所有样本的值均为正的
3. k3 （unbiased low variance estimator）: $(r-1)-\log  r$  其中 r = $\frac{\pi(s,a)}{\pi_{ref}(s,a)}$
   + 在 k1 基础上增加 $(r-1)$ 项，减小方差
   + $(r-1)$ 是无偏的
   + $(r-1)>\log r$

## ReMax

### 与 PPO 对比

ReMax 算法相比于PPO来说取消了critic model

<img src="llm.assets/2025-04-14 15-44-50屏幕截图.png" alt="2025-04-14 15-44-50屏幕截图" style="zoom:67%;" />

### 主要思路

ReMax的主要思想是去掉value function的学习和估计，直接用整个句子的奖励得分（trajectory-level reward）来计算policy gradient，与早期的 REINFORCE  算法理念一致。
$$
\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) R(\tau)
$$
作者发现仅用整个句子（轨迹）的reward优化会导致梯度十分不稳定，优化效果也不好。作者在此基础上引入baseline来减小梯度的方差。
$$
\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) (R(\tau)-b)
$$
具体地，采用当前policy 对于 给定prompt的贪心解码的completion的 奖励得分作为baseline，训练变得更加稳定同时优化效果也有明显改善。该方法实现起来也很简洁：

<img src="llm.assets/2025-04-14 15-53-27屏幕截图.png" alt="2025-04-14 15-53-27屏幕截图" style="zoom: 67%;" />

个人理解：这个算法很早就被提出来了，可能是 LLM 中第一个使用，然后就发出来了。使用 baseline 的好处是相当于提供了一个优化的方向。因为reward model给出的奖励得分有可能是恒为正的，没有baseline时假如对于同一个prompt，第一个训练轮次采样的completion奖励得分是5，第二个训练轮次采样到的completion奖励得分是1，那第二次训练时还是会朝着这个更差一些的completion的方向优化，因为它的奖励是正的。因此会给优化过程带来较大的波动。

### 回报R计算

使用 Token 级别的 KL penalty计算:
$$
r(s_t,a_t)=I(s_t=[\text{EOS}])r(x,y)-\beta \text{KL}(t)
$$

### 优势计算

使用 Token 级别的优势，对于response 中第 t 个 Token 计算公式如下：
$$
A_t=r(s_t,a_t)-r_{greedy}(s_t,a_t)
$$

## RLOO

### 与 PPO 对比

PPO这样的actor-critic算法的提出，源于传统强化学习场景中观察到的高方差问题。**作者认为，强策略初始化（例如预训练的LLM）下，方差过高并不是微调LLM时的实际问题。在这种环境下，强初始化加上提示条件，导致每步生成过程中概率质量集中于少数几个token上，尽管理论上可能的动作空间非常庞大。**因此，优化过程中不太可能出现如破坏性的大方差梯度更新之类的问题。因此，在引入偏差的情况下减少方差并不值得。

GAE引入了一个超参数 𝜆∈[0,1] ，用于平衡构建的估计器的偏差和方差。 𝜆 越接近1，方差越大。最优的 𝜆 值取决于环境。可以看到在 𝜆=1 时，偏差最小化且方差最大，结果最好。

作者比较了开启和关闭剪裁时PPO的奖励曲线。还关闭了价值网络的剪裁，因为在传统深度RL环境中，这对学习有显著影响（Engstrom等，2020）。关闭剪裁对学习几乎没有影响。在RLHF环境中发现，剪裁在每批训练中平均只影响不到5%的情况，这表明学习过程接近“on-policy”，即每次迭代之间策略变化较小。

<img src="https://pic2.zhimg.com/v2-02a5043c06c437c1513fb21ecf6f9907_1440w.jpg" alt="img" style="zoom: 50%;" />

### 主要思路

RLOO 算法思路与 ReMax差不多，一次能从策略中生成多个样本 y(1),…,y(k)，用每个样本的 reward 来帮助别的样本当 baseline。
$$
\frac{1}{k} \sum_{i=1}^{k} \left[ R(y^{(i)}, x) - \frac{1}{k-1} \sum_{\substack{j=1 \\ j \ne i}}^{k} R(y^{(j)}, x) \right] \nabla \log \pi(y^{(i)} | x)
$$
直观上可以这样理解：我写了 4 篇作文，然后互相打分，每篇作文的得分都扣掉其他3篇的平均分，再决定谁需要改进。

### 回报R计算

不是将每个token作为一个动作（即部分完成）来建模，而是将整条 response作为一个动作来建模。具体操作如下：

1. 使用 Token 级别的 KL penalty计算每个token的回报
   $$
   r(s_t,a_t)=I(s_t=[\text{EOS}])r(x,y)-\beta \text{KL}(t)
   $$
   
2. 对 response 的所有 Token的回报求和
   $$
   R=\sum_{i=1}^Tr(s_i,a_i)
   $$

### 优势计算

使用 Response 级别的优势，对于每个response，优势计算如下：
$$
A = \frac{1}{k} \sum_{i=1}^{k} \left[ R(y^{(i)}, x) - \frac{1}{k-1} \sum_{\substack{j=1 \\ j \ne i}}^{k} R(y^{(j)}, x) \right]
$$
