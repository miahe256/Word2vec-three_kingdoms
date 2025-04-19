import torch
import torch.nn as nn
from typing import List, Tuple, Dict

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
        return self.embeddings(word_idx).detach()

def load_model(model_path):
    """加载训练好的模型和词表"""
    checkpoint = torch.load(model_path)
    vocab = checkpoint['vocab']
    vector_size = checkpoint['vector_size']
    
    # 初始化模型
    model = Word2Vec(len(vocab), vector_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果有GPU则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    return model, vocab

def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    return torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item()

def get_word_vector(model, vocab, word):
    """获取词的向量表示"""
    if word not in vocab:
        raise ValueError(f"词语 '{word}' 不在词表中")
    device = next(model.parameters()).device
    word_idx = torch.tensor(vocab[word], device=device)
    return model.get_word_vector(word_idx)

def find_similar_words(model, vocab, target_word, top_k=5):
    """查找与目标词最相似的K个词"""
    if target_word not in vocab:
        raise ValueError(f"词语 '{target_word}' 不在词表中")
    
    # 获取目标词向量
    target_vector = get_word_vector(model, vocab, target_word)
    
    # 构建反向词表
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    # 计算与所有词的相似度
    similarities = []
    for word in vocab.keys():
        if word != target_word:
            vector = get_word_vector(model, vocab, word)
            similarity = cosine_similarity(target_vector, vector)
            similarities.append((word, similarity))
    
    # 按相似度排序并返回前K个
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def test_word_similarities(model, vocab, words):
    """测试词语之间的相似度"""
    for w1 in words:
        for w2 in words:
            if w1 != w2:
                try:
                    v1 = get_word_vector(model, vocab, w1)
                    v2 = get_word_vector(model, vocab, w2)
                    sim = cosine_similarity(v1, v2)
                    print(f"{w1} 和 {w2} 的相似度: {sim:.4f}")
                except ValueError as e:
                    print(e)

def word_vector_operation(model, vocab, positive_words, negative_words, top_k=5):
    """进行词向量运算，类似于 word1 + word2 - word3
    
    Args:
        model: Word2Vec模型
        vocab: 词表
        positive_words: 加法运算的词列表
        negative_words: 减法运算的词列表
        top_k: 返回最相似词的数量
    
    Returns:
        与结果向量最相似的top_k个词及其相似度
    """
    # 确保所有词都在词表中
    for word in positive_words + negative_words:
        if word not in vocab:
            raise ValueError(f"词语 '{word}' 不在词表中")
    
    # 计算正向词的向量和
    positive_vecs = [get_word_vector(model, vocab, word) for word in positive_words]
    # 计算负向词的向量和
    negative_vecs = [get_word_vector(model, vocab, word) for word in negative_words]
    
    # 计算结果向量：positive_vecs的和 - negative_vecs的和
    result_vector = sum(positive_vecs) - sum(negative_vecs)
    
    # 计算与所有词的相似度
    similarities = []
    for word in vocab.keys():
        # 排除参与运算的词
        if word not in positive_words and word not in negative_words:
            vector = get_word_vector(model, vocab, word)
            similarity = cosine_similarity(result_vector, vector)
            similarities.append((word, similarity))
    
    # 按相似度排序并返回前K个
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def analyze_character_relationships(model, vocab, character_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], str]:
    """分析人物关系
    
    Args:
        model: Word2Vec模型
        vocab: 词表
        character_pairs: 要分析的人物对列表
    
    Returns:
        人物关系字典
    """
    relationships = {}
    
    for char1, char2 in character_pairs:
        try:
            v1 = get_word_vector(model, vocab, char1)
            v2 = get_word_vector(model, vocab, char2)
            similarity = cosine_similarity(v1, v2)
            
            # 根据相似度判断关系
            if similarity >= 0.6:
                relationship = "亲密关系"  # 君臣/兄弟
            elif similarity >= 0.4:
                relationship = "友好关系"  # 同盟/好友
            elif similarity >= 0.2:
                relationship = "普通关系"  # 普通交集
            else:
                relationship = "对立关系"  # 敌对
            
            relationships[(char1, char2)] = f"{relationship} (相似度: {similarity:.4f})"
                
        except ValueError as e:
            relationships[(char1, char2)] = f"无法分析: {str(e)}"
    
    return relationships

def find_character_camp(model, vocab, character: str, camp_leaders: List[str]) -> str:
    """判断人物属于哪个阵营
    
    Args:
        model: Word2Vec模型
        vocab: 词表
        character: 要判断的人物
        camp_leaders: 阵营领袖列表 [蜀汉领袖, 魏国领袖, 东吴领袖]
    
    Returns:
        阵营判断结果
    """
    try:
        char_vector = get_word_vector(model, vocab, character)
        similarities = []
        for leader in camp_leaders:
            try:
                leader_vector = get_word_vector(model, vocab, leader)
                sim = cosine_similarity(char_vector, leader_vector)
                similarities.append((leader, sim))
            except ValueError:
                continue
        
        if similarities:
            # 找出相似度最高的领袖
            closest_leader = max(similarities, key=lambda x: x[1])
            camp_map = {
                "刘备": "蜀汉",
                "曹操": "魏国",
                "孙权": "东吴"
            }
            return f"{character} 最可能属于 {camp_map[closest_leader[0]]} 阵营 (与{closest_leader[0]}的相似度: {closest_leader[1]:.4f})"
        return f"无法判断 {character} 的阵营"
    except ValueError as e:
        return f"分析失败: {str(e)}"

def analyze_battle_keywords(model, vocab, battle_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """分析战役相关的关键词
    
    Args:
        model: Word2Vec模型
        vocab: 词表
        battle_name: 战役名称
        top_k: 返回的关键词数量
    
    Returns:
        相关关键词及其相似度列表
    """
    try:
        return find_similar_words(model, vocab, battle_name, top_k)
    except ValueError as e:
        return []

def predict_character_behavior(model, vocab, character: str, behavior_categories: Dict[str, List[str]]) -> str:
    """预测人物可能的行为类型
    
    Args:
        model: Word2Vec模型
        vocab: 词表
        character: 人物名称
        behavior_categories: 行为类别字典
    
    Returns:
        预测的行为类型
    """
    try:
        char_vector = get_word_vector(model, vocab, character)
        category_scores = {}
        
        for category, behaviors in behavior_categories.items():
            category_score = 0
            valid_behaviors = 0
            for behavior in behaviors:
                try:
                    behavior_vector = get_word_vector(model, vocab, behavior)
                    sim = cosine_similarity(char_vector, behavior_vector)
                    category_score += sim
                    valid_behaviors += 1
                except ValueError:
                    continue
            
            if valid_behaviors > 0:
                category_scores[category] = category_score / valid_behaviors
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            return f"{character} 最可能的行为类型是 {best_category[0]} (匹配度: {best_category[1]:.4f})"
        return f"无法预测 {character} 的行为类型"
    except ValueError as e:
        return f"预测失败: {str(e)}"

def generate_character_dialogue(model, vocab, context: str, character: str, top_k: int = 3) -> List[str]:
    """根据上下文生成人物可能的对话
    
    Args:
        model: Word2Vec模型
        vocab: 词表
        context: 上下文场景
        character: 人物名称
        top_k: 生成的对话数量
    
    Returns:
        可能的对话列表
    """
    try:
        # 将上下文和人物结合
        combined_vector = get_word_vector(model, vocab, context)
        char_vector = get_word_vector(model, vocab, character)
        
        # 结合向量
        combined = combined_vector + char_vector
        
        # 找到最相关的词语组合
        similarities = []
        for word in vocab.keys():
            if word != context and word != character:
                vector = get_word_vector(model, vocab, word)
                similarity = cosine_similarity(combined, vector)
                similarities.append((word, similarity))
        
        # 按相似度排序并返回前K个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [f'{character}说："{word}" (匹配度: {sim:.4f})' 
                for word, sim in similarities[:top_k]]
    except ValueError as e:
        return [f"生成失败: {str(e)}"]

def analyze_emotion(model, vocab, text: str, emotion_words: Dict[str, List[str]]) -> str:
    """分析文本情感倾向
    
    Args:
        model: Word2Vec模型
        vocab: 词表
        text: 要分析的文本
        emotion_words: 情感词字典
    
    Returns:
        情感分析结果
    """
    try:
        text_vector = get_word_vector(model, vocab, text)
        emotion_scores = {}
        
        for emotion, words in emotion_words.items():
            emotion_score = 0
            valid_words = 0
            for word in words:
                try:
                    word_vector = get_word_vector(model, vocab, word)
                    sim = cosine_similarity(text_vector, word_vector)
                    emotion_score += sim
                    valid_words += 1
                except ValueError:
                    continue
            
            if valid_words > 0:
                emotion_scores[emotion] = emotion_score / valid_words
        
        if emotion_scores:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            return f"文本 '{text}' 的情感倾向是 {dominant_emotion[0]} (置信度: {dominant_emotion[1]:.4f})"
        return f"无法分析文本 '{text}' 的情感倾向"
    except ValueError as e:
        return f"分析失败: {str(e)}"

def build_knowledge_graph(model, vocab, relationships: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
    """构建简单的知识图谱
    
    Args:
        model: Word2Vec模型
        vocab: 词表
        relationships: 关系三元组列表 [(实体1, 关系, 实体2)]
    
    Returns:
        知识图谱中的关系列表
    """
    graph = []
    for entity1, relation, entity2 in relationships:
        try:
            v1 = get_word_vector(model, vocab, entity1)
            v2 = get_word_vector(model, vocab, entity2)
            similarity = cosine_similarity(v1, v2)
            
            graph.append({
                "entity1": entity1,
                "relation": relation,
                "entity2": entity2,
                "confidence": similarity
            })
        except ValueError as e:
            continue
    
    return graph

if __name__ == "__main__":
    # 加载模型
    model_path = './models/word2vec_torch_three_kingdoms.pt'
    model, vocab = load_model(model_path)
    
    # 测试词语相似度
    test_words = ['曹操', '刘备', '孔明', '关羽']
    print("\n（1）测试词语相似度:")
    test_word_similarities(model, vocab, test_words)
    
    # 查找相似词
    print("\n（2）查找相似词:")
    for word in test_words:
        try:
            similar_words = find_similar_words(model, vocab, word, top_k=5)
            print(f"\n与 '{word}' 最相似的词语:")
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        except ValueError as e:
            print(e)
    
    # 测试词向量运算
    print("\n（3）测试词向量运算:")
    try:
        # 计算 刘备 + 孔明 - 曹操
        result = word_vector_operation(
            model, 
            vocab,
            positive_words=['刘备', '曹操'],
            negative_words=['张飞'],
            top_k=5
        )
        print("\n'曹操+刘备-张飞' 最相似的词语:")
        for word, similarity in result:
            print(f"  {word}: {similarity:.4f}")
    except ValueError as e:
        print(e)
    
    # 新增场景测试
    print("\n（4）人物关系分析:")
    character_pairs = [
        ("刘备", "诸葛亮"),
        ("关羽", "张飞"),
        ("曹操", "刘备"),
        ("孙权", "周瑜")
    ]
    relationships = analyze_character_relationships(model, vocab, character_pairs)
    for (char1, char2), relation in relationships.items():
        print(f"{char1} 与 {char2} 的关系: {relation}")
    
    print("\n（5）阵营分析:")
    characters = ["诸葛亮", "司马懿", "周瑜", "关羽"]
    camp_leaders = ["刘备", "曹操", "孙权"]
    for character in characters:
        result = find_character_camp(model, vocab, character, camp_leaders)
        print(result)
    
    print("\n（6）战役关键词分析:")
    battles = ["赤壁之战", "官渡之战"]
    for battle in battles:
        keywords = analyze_battle_keywords(model, vocab, battle)
        if keywords:
            print(f"\n{battle}的相关关键词:")
            for word, similarity in keywords:
                print(f"  {word}: {similarity:.4f}")
    
    print("\n（7）人物行为预测:")
    behavior_categories = {
        "智谋": ["计策", "谋略", "智取"],
        "武力": ["征战", "厮杀", "战斗"],
        "仁德": ["仁义", "德行", "宽厚"]
    }
    characters = ["诸葛亮", "关羽", "刘备"]
    for character in characters:
        result = predict_character_behavior(model, vocab, character, behavior_categories)
        print(result)
    
    print("\n（8）场景对话生成:")
    contexts = ["战场", "朝堂"]
    characters = ["诸葛亮", "曹操"]
    for context in contexts:
        for character in characters:
            print(f"\n在{context}场景下，{character}可能说的话:")
            dialogues = generate_character_dialogue(model, vocab, context, character)
            for dialogue in dialogues:
                print(dialogue)
    
    print("\n（9）情感分析:")
    emotion_words = {
        "正面": ["忠义", "仁德", "智谋"],
        "负面": ["奸诈", "背叛", "暴虐"]
    }
    texts = ["忠义千秋", "奸诈小人"]
    for text in texts:
        result = analyze_emotion(model, vocab, text, emotion_words)
        print(result)
    
    print("\n（10）知识图谱构建:")
    relationships = [
        ("刘备", "结义兄弟", "关羽"),
        ("诸葛亮", "军师", "刘备"),
        ("曹操", "宿敌", "刘备")
    ]
    graph = build_knowledge_graph(model, vocab, relationships)
    print("\n知识图谱关系:")
    for relation in graph:
        print(f"{relation['entity1']} -{relation['relation']}-> {relation['entity2']} "
              f"(置信度: {relation['confidence']:.4f})") 