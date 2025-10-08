import pickle

# 定义文件路径
file_path = 'datasets/pdp_20.pkl'

# 使用 'rb' 模式打开文件，表示以二进制读取模式
with open(file_path, 'rb') as f:
    # 使用 pickle.load() 来反序列化文件内容
    data = pickle.load(f)



# 检查数据是否为列表或元组，然后打印前两个元素
if isinstance(data, (list, tuple)):
    print(f"总共有 {len(data)} 个数据点")
    # 打印前两个数据点
    for i, item in enumerate(data[:2]):
        print(f"数据 {i+1}:\n{item}\n")
else:
    # 如果数据不是列表或元-组，则打印数据类型并尝试打印整个数据
    print(f"数据类型为: {type(data)}")
    print("数据内容（或部分内容）:")
    # 尝试打印，但如果数据结构复杂，可能仍然会输出大量信息
    print(data)