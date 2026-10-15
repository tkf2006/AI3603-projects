"""
GPT-2 风格小模型框架模板
=========================

本模板只提供必要的类名、函数名和伪代码提示。
你需要补全所有 `pass` 处的实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import math


# ============================================================
# 1. 配置
# ============================================================
@dataclass
class GPTConfig:
    block_size: int = 512
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    vocab_size: int = 50257

    # 提示：head_size 可以不显式存储，在 Attention 内部计算即可


# ============================================================
# 2. 单头注意力
# ============================================================
class SingleHeadAttention(nn.Module):
    """
    单头因果自注意力。

    输入: x, shape (B, T, C)
    输出: out, shape (B, T, head_size)
    """
    def __init__(self, config):
        super().__init__()
        # 提示：
        #   self.head_size = config.n_embd // config.n_head
        #   定义 self.query, self.key, self.value 三个线性层
        #   定义因果掩码 buffer（下三角矩阵）
        #   定义 dropout
        pass

    def forward(self, x):
        # 提示：
        #   B, T, C = x.shape
        #   q = self.query(x)  # (B, T, head_size)
        #   k = self.key(x)
        #   v = self.value(x)
        #   wei = q @ k.transpose(-2, -1) / sqrt(head_size)  # (B, T, T)
        #   wei = wei.masked_fill(mask[:T, :T] == 0, -inf)
        #   wei = softmax(wei, dim=-1)
        #   wei = dropout(wei)
        #   out = wei @ v  # (B, T, head_size)
        #   return out
        pass


# ============================================================
# 3. 多头注意力
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力：把 n_head 个 SingleHeadAttention 的输出拼接后投影。
    """
    def __init__(self, config):
        super().__init__()
        # 提示：
        #   self.heads = nn.ModuleList([SingleHeadAttention(config)
        #                               for _ in range(config.n_head)])
        #   self.proj = nn.Linear(config.n_embd, config.n_embd)
        #   self.dropout = nn.Dropout(config.dropout)
        pass

    def forward(self, x):
        # 提示：
        #   out = torch.cat([h(x) for h in self.heads], dim=-1)
        #   out = self.proj(out)
        #   out = self.dropout(out)
        #   return out
        pass


# ============================================================
# 4. 前馈网络
# ============================================================
class FeedForward(nn.Module):
    """
    MLP: Linear -> GELU -> Linear -> Dropout
    隐藏维度通常是 n_embd 的 4 倍。
    """
    def __init__(self, config):
        super().__init__()
        # 提示：
        #   self.net = nn.Sequential(
        #       nn.Linear(config.n_embd, 4 * config.n_embd),
        #       nn.GELU(),
        #       nn.Linear(4 * config.n_embd, config.n_embd),
        #       nn.Dropout(config.dropout),
        #   )
        pass

    def forward(self, x):
        # 提示：return self.net(x)
        pass


# ============================================================
# 5. Transformer Block
# ============================================================
class Block(nn.Module):
    """
    Pre-norm 结构：
        x = x + attn(ln1(x))
        x = x + ffn(ln2(x))
    """
    def __init__(self, config):
        super().__init__()
        # 提示：
        #   self.ln1 = nn.LayerNorm(config.n_embd)
        #   self.att = MultiHeadAttention(config)
        #   self.ln2 = nn.LayerNorm(config.n_embd)
        #   self.ffn = FeedForward(config)
        pass

    def forward(self, x):
        # 提示：
        #   x = x + self.att(self.ln1(x))
        #   x = x + self.ffn(self.ln2(x))
        #   return x
        pass


# ============================================================
# 6. GPT 主模型
# ============================================================
class GPT(nn.Module):
    """
    GPT-2 风格的 decoder-only Transformer。
    """
    def __init__(self, config):
        super().__init__()
        # 提示：
        #   self.config = config
        #   self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #   self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #   self.blocks = nn.Sequential(*[Block(config) for _ in range(n_layer)])
        #   self.ln_final = nn.LayerNorm(n_embd)
        #   self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        #   权重初始化：Linear 用 N(0, 0.02)，bias 置 0，Embedding 用 N(0, 0.02)
        pass

    def forward(self, idx, targets=None):
        """
        idx: (B, T) int64
        targets: (B, T) int64 或 None
        返回: logits (B, T, vocab_size), loss 或 None
        """
        # 提示：
        #   B, T = idx.shape
        #   tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        #   pos = torch.arange(T, device=idx.device)
        #   pos_emb = self.position_embedding_table(pos)  # (T, C)
        #   x = tok_emb + pos_emb
        #   x = self.blocks(x)
        #   x = self.ln_final(x)
        #   logits = self.lm_head(x)  # (B, T, vocab_size)
        #
        #   if targets is None:
        #       loss = None
        #   else:
        #       logits_flat = logits.view(B*T, vocab_size)
        #       targets_flat = targets.view(B*T)
        #       loss = F.cross_entropy(logits_flat, targets_flat)
        #   return logits, loss
        pass

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, block_size):
        """
        Greedy decoding。
        idx: (B, T) 初始 prompt
        返回: (B, T + max_new_tokens)
        """
        # 提示：
        #   for _ in range(max_new_tokens):
        #       idx_cond = idx[:, -block_size:]
        #       logits, _ = self(idx_cond)
        #       logits = logits[:, -1, :]  # (B, vocab_size)
        #       next_id = argmax(logits, dim=-1, keepdim=True)
        #       idx = torch.cat([idx, next_id], dim=1)
        #   return idx
        pass


# ============================================================
# 7. 数据集
# ============================================================
class MyDataset(Dataset):
    """
    从 TinyStories txt 文件读取故事，切分成 block_size 长度的训练样本。

    数据格式：每个故事以 <|endoftext|> 分隔。
    """
    def __init__(self, path, tokenizer, block_size=512, max_lines=None):
        """
        path: txt 文件路径
        tokenizer: 你的 BPE 分词器对象，需要提供 encode(text) -> List[int]
        block_size: 每条样本的长度
        max_lines: 读取的故事数上限，None 表示全部
        """
        # 提示：
        #   1. 读文件，按 <|endoftext|> 切分故事
        #   2. 逐条编码，加 eos token
        #   3. 拼成一条长序列，按 block_size + 1 切块
        #   4. 每块 x = chunk[:-1], y = chunk[1:]
        #   建议用 numpy.int32 存储，避免内存爆炸
        pass

    def __len__(self):
        # 提示：返回样本数
        pass

    def __getitem__(self, idx):
        # 提示：返回 (x, y)，都是 torch.long
        pass


# ============================================================
# 8. 训练函数
# ============================================================
def train(model, optimizer, scheduler, train_loader, device, epoch,
          log_every=100, scaler=None):
    """
    训练一个 epoch。
    返回: (total_loss, steps_done)
    """
    model.train()
    # 提示：
    #   for batch_idx, (x, y) in enumerate(train_loader):
    #       x, y = x.to(device), y.to(device)
    #       if scaler:
    #           with autocast(dtype=torch.float16):
    #               logits, loss = model(x, targets=y)
    #           optimizer.zero_grad()
    #           scaler.scale(loss).backward()
    #           scaler.unscale_(optimizer)
    #           clip_grad_norm_(model.parameters(), 1.0)
    #           scaler.step(optimizer)
    #           scaler.update()
    #       else:
    #           logits, loss = model(x, targets=y)
    #           optimizer.zero_grad()
    #           loss.backward()
    #           optimizer.step()
    #       scheduler.step()
    #       total_loss += loss.item()
    #       if batch_idx % log_every == 0:
    #           print(...)
    pass


# ============================================================
# 9. 评估函数
# ============================================================
@torch.no_grad()
def evaluate(model, val_loader, device):
    """
    在验证集上计算平均 loss。
    返回: avg_loss (float)
    """
    model.eval()
    # 提示：
    #   total_loss, total_tokens = 0.0, 0
    #   for x, y in val_loader:
    #       x, y = x.to(device), y.to(device)
    #       logits, loss = model(x, targets=y)
    #       total_loss += loss.item() * y.numel()
    #       total_tokens += y.numel()
    #   return total_loss / total_tokens
    pass


# ============================================================
# 10. BPB 计算工具
# ============================================================
def compute_bpb(avg_loss_nats, token_count, byte_count):
    """
    avg_loss_nats: 验证集上的平均交叉熵损失（自然对数）
    token_count: 验证集 token 总数
    byte_count: 验证集原始文本的 UTF-8 字节数
    """
    # 提示：
    #   total_nats = avg_loss_nats * token_count
    #   bpb = total_nats / (byte_count * math.log(2))
    #   return bpb
    pass


# ============================================================
# 11. 主训练脚本（示例）
# ============================================================
def main():
    # 提示：
    #   1. 设置随机种子
    #   2. 加载 tokenizer（你自己训练的 BPE）
    #   3. 构建 train_dataset / val_dataset / DataLoader
    #   4. 构建 GPTConfig 和 GPT 模型
    #   5. 定义 optimizer (AdamW, lr=3e-4, betas=(0.9, 0.95))
    #   6. 定义 scheduler (CosineAnnealingLR 或 OneCycleLR)
    #   7. 定义 GradScaler（FP16）
    #   8. 训练循环：
    #       for epoch in range(num_epochs):
    #           train(...)
    #           val_loss = evaluate(...)
    #           bpb = compute_bpb(val_loss, ...)
    #           if val_loss < best_val_loss:
    #               save checkpoint
    #   9. 生成样本并保存
    pass


if __name__ == "__main__":
    main()