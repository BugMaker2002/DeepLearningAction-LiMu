import jieba
text = "结巴分词是一款开源的中文分词工具。"
words = jieba.cut(text)

# 将分词结果转换为列表
word_list = list(words)
print(word_list[0][-1])
