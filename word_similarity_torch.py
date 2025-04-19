# -*-coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import os

class Word2VecDataset(Dataset):
    def __init__(self, segment_folder, window_size=3, min_count=1):
        # 读取所有分词后的文件
        self.sentences = []
        for file in os.listdir(segment_folder):
            if file.startswith('segment_') and file.endswith('.txt'):
                with open(os.path.join(segment_folder, file), 'rb') as f:
                    content = f.read().decode('utf-8')
                    self.sentences.extend(content.strip().split())
        
        # 构建词表
        word_counts = Counter(self.sentences)
        self.vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items()) 
                     if count >= min_count}
        self.vocab_size = len(self.vocab)
        
        # 构建训练数据
        self.window_size = window_size
        self.data = []
        for i in range(len(self.sentences)):
            for j in range(max(0, i - window_size), min(len(self.sentences), i + window_size + 1)):
                if i != j and self.sentences[i] in self.vocab and self.sentences[j] in self.vocab:
                    self.data.append((self.sentences[i], self.sentences[j]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target_word, context_word = self.data[idx]
        target_idx = self.vocab[target_word]
        context_idx = self.vocab[context_word]
        return torch.tensor(target_idx), torch.tensor(context_idx)

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out

    def get_word_vector(self, word_idx):
        # word_idx 已经在正确的设备上，直接使用
        return self.embeddings(word_idx).detach()

def train_word2vec(segment_folder, vector_size=100, window=3, min_count=1, epochs=5):
    # 准备数据
    print("正在加载数据集...")
    dataset = Word2VecDataset(segment_folder, window, min_count)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(f"数据集加载完成，词表大小: {dataset.vocab_size}，训练样本数: {len(dataset)}")
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = Word2Vec(dataset.vocab_size, vector_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # 训练模型
    print("\n开始训练...")
    for epoch in range(epochs):
        total_loss = 0
        batch_count = len(dataloader)
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for batch_idx, (target, context) in enumerate(dataloader):
            target, context = target.to(device), context.to(device)
            optimizer.zero_grad()
            output = model(target)
            loss = criterion(output, context)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 显示进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == batch_count:
                progress = (batch_idx + 1) / batch_count * 100
                print(f"\r进度: [{batch_idx + 1}/{batch_count}] {progress:.2f}% "
                      f"当前损失: {loss.item():.4f}", end="")
        
        epoch_loss = total_loss/len(dataloader)
        print(f"\nEpoch {epoch+1} 完成, 平均损失: {epoch_loss:.4f}")
    
    print("\n训练完成!")
    return model, dataset.vocab

def cosine_similarity(v1, v2):
    return torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item()

if __name__ == "__main__":
    # 训练模型
    segment_folder = './three_kingdoms/segment'
    model, vocab = train_word2vec(segment_folder, vector_size=100, window=3, min_count=1)
    
    # 构建反向词表
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    # 计算相似度
    def get_word_vector(word):
        if word not in vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")
        # 获取模型所在的设备
        device = next(model.parameters()).device
        # 确保输入tensor在正确的设备上
        word_idx = torch.tensor(vocab[word], device=device)
        return model.get_word_vector(word_idx)
    
    # 测试相似度
    words = ['曹操', '刘备', '孔明', '关羽']
    for w1 in words:
        for w2 in words:
            if w1 != w2:
                try:
                    v1 = get_word_vector(w1)
                    v2 = get_word_vector(w2)
                    sim = cosine_similarity(v1, v2)
                    print(f"{w1} 和 {w2} 的相似度: {sim:.4f}")
                except ValueError as e:
                    print(e)
    
    # 确保models目录存在
    os.makedirs('./models', exist_ok=True)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'vector_size': 100
    }, './models/word2vec_torch_three_kingdoms.pt') 