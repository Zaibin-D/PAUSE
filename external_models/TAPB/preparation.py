import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, EsmModel
from utils.utils import load_config_file, mkdir

# 所有这个generate_esm2_feature整体作用就是读取csv里面的内容，将每个蛋白质都经过esm2然后得到对应的有语义的向量
def generate_esm2_feature(config, dataset, split):
    # 打印提示信息，告诉用户程序开始运行了
    print('start generating esm2 feature')

    # 1. 准备路径
    # 根据数据集名字（如 biosnap）和切分方式（如 random）拼接出数据文件夹路径
    dataset_path = f'datasets/{dataset}/{split}/'
    # 如果文件夹不存在，就创建一个（防止报错）
    mkdir(dataset_path)

    # 2. 选择设备
    # 自动检测：如果有显卡 (cuda:0) 就用显卡，否则用 CPU。ESM-2 模型很大，强烈建议用显卡。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3. 读取并合并数据
    # 根据切分方式（split）读取不同的 CSV 文件
    if split == 'cluster':
        # 如果是聚类切分（通常用于跨域测试），文件名可能叫 source_train 等
        dfs = [pd.read_csv(f'{dataset_path}{dataset}_with_id.csv') for dataset in
               ['source_train', 'target_train', 'target_test']]
    else:
        # 如果是普通切分（In-domain），读取训练、验证、测试三个集
        dfs = [pd.read_csv(f'{dataset_path}{dataset}_with_id.csv') for dataset in ['train', 'val', 'test']]

    # 将读取到的所有数据表拼成一个巨大的表格 (DataFrame)，方便统一处理
    df = pd.concat(dfs)

    # 4. 加载 ESM-2 模型
    # 指定你之前下载好的 ESM-2 权重文件夹路径
    ems2_model_path = 'models/protein/esm2_model'
    # 加载分词器（负责把氨基酸字母变成数字）
    tokenizer = AutoTokenizer.from_pretrained(ems2_model_path)
    # 加载模型本体，并搬运到显卡上 (to device)
    model = EsmModel.from_pretrained(ems2_model_path).to(device)
    # 开启“评估模式”。这会关闭 Dropout，保证每次算出来的特征都是固定的，不会变。
    model.eval()

    # 创建一个空列表，准备存放所有提取出来的特征
    prlist = list()

    # 5. 核心循环：逐个处理蛋白质
    # df['pr_id'].unique(): 这一步非常关键！它对蛋白质 ID 进行了去重。
    # 因为训练集里可能有 1000 条数据都用了同一个蛋白，我们只需要算一次特征就够了，不要重复算。
    for protein_id in tqdm(df['pr_id'].unique(), desc='Processing'):
        # 根据 ID 找到对应的氨基酸序列（Protein列）。iloc[0] 取第一条即可。
        protein_seq = df[df['pr_id'] == protein_id]['Protein'].iloc[0]

        # 5.1 数据预处理 (Tokenize)
        # 将文本序列转为模型能读懂的 Tensor。
        # truncation=True, max_length=2000: 如果序列超过2000个氨基酸，强行截断。
        # 这是为了防止显存爆炸（OOM），因为 Transformer 对长序列的显存占用是平方级增长的。
        inputs = tokenizer(protein_seq, return_tensors="pt", truncation=True, max_length=2000).to(device)

        # 5.2 模型推理 (Inference)
        # torch.no_grad(): 告诉 PyTorch 不要计算梯度。我们只是用模型，不训练模型。
        # 这能极大节省显存，并加快速度。
        with torch.no_grad():
            outputs = model(**inputs)

        # 5.3 提取特征
        # last_hidden_state: 获取模型最后一层的输出 (Shape: [1, 序列长度, 维度])
        # squeeze(): 把第0维的 batch 维度去掉，变成 [序列长度, 维度]
        sr = outputs.last_hidden_state.squeeze()

        # 将提取好的特征 Tensor 加入到列表中
        # 注意：这里存的是 Tensor，为了节省内存，通常应该把它转回 CPU (.cpu()) 再存，原代码可能因为数据量不大直接存了。
        prlist.append(sr)

    # 6. 保存结果
    # 拼接保存文件的完整路径（通常在 config 里配置了文件名，如 'esm2_feat.pkl'）
    save_path = os.path.join(dataset_path, config.TRAIN.PR_PATH)

    # 以“二进制写入模式” (wb) 打开文件
    file = open(save_path, 'wb')

    # 使用 pickle 库将整个列表序列化并保存到硬盘上
    pickle.dump(prlist, file)

    # 打印结束信息
    print('finish generating esm2 feature')


def kmeans_c(featurelist, config, d_model):
    # 1. 数据准备
    # featurelist 是一个列表，里面装着 N 个 [1280] 维的向量 这里的N其实就是蛋白质的数量
    # vstack 把它们垂直堆叠成一个巨大的矩阵，形状变为 [N, 1280]
    matrix = np.vstack(featurelist)

    # 2. 初始化 K-Means 模型
    # n_clusters 读取配置里的字典大小 (DICT_SIZE)，即论文中的 K (例如 4 或 8)
    kmeans = KMeans(n_clusters=config.TRAIN.DICT_SIZE)

    # 3. 执行聚类
    # 让 K-Means 算法在矩阵上跑，自动找到 K 个聚类中心
    kmeans.fit(matrix)

    # 4. 获取标签
    # 拿到每个样本属于哪个簇的标签 (例如 [0, 1, 0, 3, ...])
    labels = kmeans.labels_

    # 5. 初始化中心矩阵
    # 创建一个全零矩阵来存放结果。
    # 注意形状是 [特征维度, 簇的数量]，例如 [1280, 4]。这意味着每一列是一个聚类中心。
    cluster_centers = np.zeros((d_model, config.TRAIN.DICT_SIZE))

    # 6. 初始化概率列表
    prior = list()  # 用来存每个簇里有多少个样本
    total = 0  # 用来统计总样本数

    # 7. 循环计算每个簇的中心和先验概率
    for i in range(config.TRAIN.DICT_SIZE):
        # matrix[labels == i]: 这是一个布尔索引
        # 它的作用是：把所有被 K-Means 归类为第 i 类的蛋白质向量全挑出来
        cluster_position = matrix[labels == i]

        # 统计第 i 类一共有多少个样本
        num = cluster_position.shape[0]

        # 记录数量
        prior.append(num)
        total += num  # 累加总数

        # 核心计算：计算这一类所有向量的平均值 (Mean)
        # axis=0 表示沿着列的方向求平均，得到一个 [1280] 的平均向量
        # cluster_centers[:, i] 表示把这个平均向量填入第 i 列
        cluster_centers[:, i] = np.mean(cluster_position, axis=0)

    # 8. 计算先验概率 P(c)
    # 将列表转为 PyTorch Tensor
    prior = torch.tensor(prior)
    # 归一化：每个类的数量除以总数，得到概率。例如 [0.2, 0.3, 0.1, 0.4]
    prior = prior / total

    # 9. 打包结果
    cluster_dict = dict()
    cluster_dict['cluster_centers'] = cluster_centers  # 存入中心特征矩阵
    cluster_dict['prior'] = prior  # 存入先验概率分布

    # 返回字典
    return cluster_dict


def kmeans_for_c(train_config, train_df, save_path):
    print('start generating confounder dict & aa dict')  # 打印开始信息

    # 1. 设备与环境配置
    # 自动选择显卡 (cuda:0) 或 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()  # 清空显存，防止之前的操作残留垃圾数据导致 OOM
    df = train_df  # 拿到训练集数据

    # 2. 加载 ESM-2 模型 (用于提取特征)
    ems2_model_path = 'models/protein/esm2_model'
    max_length = 2000  # 序列最大长度限制
    d_model = 1280  # ESM-2 模型的输出维度 (t33版本通常是1280)

    # 加载分词器和模型，并移动到 GPU
    tokenizer = AutoTokenizer.from_pretrained(ems2_model_path)
    model = EsmModel.from_pretrained(ems2_model_path).to(device)
    model.eval()  # 开启评估模式，关闭 Dropout，保证结果确定性

    # 3. 初始化容器
    featurelist = list()  # 用于存放每个蛋白质的“整体平均特征” (用于聚类)
    aa_features = defaultdict(list)  # 字典：key是氨基酸字母('A','M'...), value是该字母出现过的所有向量列表

    # 4. 核心循环：遍历训练集中的每一个蛋白质
    for protein_id in tqdm(df['pr_id'].unique(), desc='Processing'):
        # 获取该 ID 对应的氨基酸序列字符串
        protein_seq = df[df['pr_id'] == protein_id]['Protein'].iloc[0]

        # 4.1 预处理与推理
        # 将序列转为 Tensor，截断长度，搬到 GPU
        inputs = tokenizer(protein_seq, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():  # 不计算梯度，节省显存
            outputs = model(**inputs)

        # sr 是序列特征，形状 [Sequence_Length + 2, 1280] (因为有头尾 Token)
        sr = outputs.last_hidden_state.squeeze()

        # 4.2 任务一：为了聚类 (计算 confounder C)
        # 对整个序列取平均 (mean(0))，得到一个 [1280] 的向量，代表这整个蛋白质的语义
        # 存入 featurelist，后续用来做 K-Means
        featurelist.append(sr.mean(0).cpu())

        # 4.3 任务二：为了氨基酸突变 (构建 AA 字典)
        # sr[1:-1] 去掉头部的 [CLS] 和尾部的 [EOS]，只保留真实的氨基酸特征
        sr_2 = sr[1:-1, :].cpu()

        # 遍历序列中的每一个氨基酸
        for i in range(sr_2.size(0)):
            aa = protein_seq[i]  # 获取当前位置的字符 (比如 'M')
            vector = sr_2[i]  # 获取当前位置的特征向量
            aa_features[aa].append(vector)  # 把这个向量扔进对应的列表中

    # 5. 计算每种氨基酸的“标准画像” (平均向量)
    aa_avg_list = []
    # sorted(aa_features.keys()) 保证顺序固定 (如 A, C, D, E...)
    for aa in sorted(aa_features.keys()):
        vectors = torch.stack(aa_features[aa])  # 把该字母收集到的所有向量堆叠成 Tensor
        avg_vector = torch.mean(vectors, dim=0)  # 算平均值 -> 得到该氨基酸的“标准向量”
        aa_avg_list.append(avg_vector)

    # 将所有氨基酸的平均向量堆叠成一个矩阵 [20种氨基酸, 1280]
    aa_tensor = torch.stack(aa_avg_list)

    # 6. 执行 K-Means 聚类
    # 调用外部函数 kmeans_c，传入所有蛋白质的整体特征，算出 K 个聚类中心 (混杂因子)
    cluster_dict = kmeans_c(featurelist, train_config, d_model)

    # 7. 打包保存
    cluster_dict['aa'] = aa_tensor  # 把氨基酸字典也塞进去

    # 保存为 pickle 文件 (对应 config 里的 C_PATH)
    file = open(save_path + '/' + train_config.TRAIN.C_PATH, 'wb')
    pickle.dump(cluster_dict, file)

    torch.cuda.empty_cache()  # 清理
    print('finish generating confounder dict & aa dict')
