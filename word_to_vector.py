# 导入numpy库
import numpy as np

# 定义一个字母表，包含26个英文字母
alphabet = "abcdefghijklmnopqrstuvwxyz"


# 定义一个函数，将一个字母转换为一个长度为26的向量，其中只有对应位置为1，其他位置为0
def letter_to_vector(letter):
    # 创建一个全零的向量
    vector = np.zeros(26)

    # 找到字母在字母表中的索引
    index = alphabet.index(letter)

    # 将对应位置设为1
    vector[index] = 1

    # 返回向量
    return vector


# 定义一个函数，将一个单词转换为一个长度为130的向量，即将每个字母的向量拼接起来
def word_to_vector(word):
    # 创建一个空列表，用于存储每个字母的向量
    vectors = []

    # 对每个字母进行循环
    for letter in word:
        # 调用上面定义的函数，将字母转换为向量，并添加到列表中
        vectors.append(letter_to_vector(letter))

    # 将列表中的所有向量拼接起来，形成一个大向量，并返回
    return np.concatenate(vectors)


# 测试一下函数是否正确工作

# 定义一个五个字母的单词
word = "hello"

# 调用函数，将单词转换为向量，并打印结果
vector = word_to_vector(word)
print("Word:", word)
print("Vector:", vector)