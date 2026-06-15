import argparse
import os
import pickle
import warnings
from time import time
import pandas as pd
import torch
from omegaconf import OmegaConf
from dataloader.dataloader import DTIDataset, get_dataLoader
from transformers import AutoTokenizer
from models.tapb import TAPB
from trainer import Trainer
from utils.utils import set_seed, mkdir, load_config_file
from preparation import generate_esm2_feature, kmeans_for_c

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="TAPB for DTI prediction")
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task",
                    choices=['random', 'cold', 'cluster', 'augmented'])
args = parser.parse_args()

TRAIN_CONFIG_PATH = 'configs/train_config.yaml'
MODEL_CONFIG_PATH = 'configs/model_config.yaml'
print(f"Running on: {device}", end="\n\n")

def main():
    # 1. 清理与设置
    torch.cuda.empty_cache()  # 清空显存，防止之前的残留数据导致 OOM (Out Of Memory)
    # 忽略“除以零”产生的无效值警告，保持控制台清爽
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    # 2. 加载配置
    train_config = load_config_file(TRAIN_CONFIG_PATH)  # 加载训练配置 (如 batch_size, lr)
    model_config = load_config_file(MODEL_CONFIG_PATH)  # 加载模型配置 (如 hidden_dim, layers)
    config = OmegaConf.merge(train_config, model_config)  # 合并两个配置到一个对象里
    model_configs = dict(model_config)  # 将模型配置转为字典格式，方便后续传参

    # 3. 确定输出路径与随机种子
    set_seed(seed=config.TRAIN.SEED)  # 设置随机种子，保证实验结果可复现
    # 拼接结果保存路径，格式如 "./results/biosnap/random/output_seed42"
    output_path = f"./results/{args.data}/{args.split}/{config.TRAIN.OUTPUT_DIR}{config.TRAIN.SEED}"
    mkdir(output_path)  # 创建该目录

    # 4. 确定数据路径
    dataFolder = f'./datasets/{args.data}'  # 基础数据目录，如 ./datasets/biosnap
    dataFolder = os.path.join(dataFolder, str(args.split))  # 加上切分方式，如 ./datasets/biosnap/random
    mol_path = 'models/drug/molformer'  # 指定 Molformer 预训练权重所在的文件夹

    # 5. 数据集划分逻辑 (In-domain vs Cross-domain)
    if args.split == 'cluster':
        # 如果是 'cluster' (跨域/聚类切分)，文件名通常不同
        train_path = os.path.join(dataFolder, 'source_train_with_id.csv')  # 源域训练集
        val_path = os.path.join(dataFolder, "target_train_with_id.csv")  # 目标域训练集作为验证
        test_path = os.path.join(dataFolder, "target_test_with_id.csv")  # 目标域测试集
    else:
        # 如果是 'random' (同域/随机切分)，读取标准的 train/val/test
        train_path = os.path.join(dataFolder, 'train_with_id.csv')
        val_path = os.path.join(dataFolder, "val_with_id.csv")
        test_path = os.path.join(dataFolder, "test_with_id.csv")

    # 6. 读取 CSV 文件到 Pandas DataFrame
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 7. 检查并生成预处理文件 (核心检查点！)
    protein_path = os.path.join(dataFolder, config.TRAIN.PR_PATH)  # ESM-2 特征文件路径 (.pkl)
    c_path = os.path.join(dataFolder, config.TRAIN.C_PATH)  # 混杂字典文件路径 (.pkl)

    # 如果 ESM-2 特征文件不存在，立刻调用函数生成它！(这就是为什么你不用手动跑 preparation.py)
    if not os.path.isfile(protein_path):
        generate_esm2_feature(config, args.data, args.split)

    # 如果混杂字典文件不存在，立刻调用函数生成它！
    if not os.path.isfile(c_path):
        kmeans_for_c(config, df_train, dataFolder)

    # 8. 加载混杂字典数据
    C = pickle.load(open(c_path, 'rb'))  # 读取生成的字典文件

    # 提取混杂中心 (Cluster Centers)，转为 Tensor 并移动到 GPU
    # permute(1, 0) 是为了调整维度，适应模型内部的矩阵乘法
    c = torch.from_numpy(C['cluster_centers']).to(device).permute(1, 0).to(dtype=torch.float32)

    # 提取先验概率 P(c_i)，用于后门调整
    p_ci = C['prior'].to(device).to(dtype=torch.float32)

    # 提取氨基酸标准特征字典，用于随机突变 (Mutation)
    aa_dict = C['aa'].to(device).to(dtype=torch.float32)

    # 9. 初始化模型 (TAPB-Full)
    # 将计算好的 c, p_ci 传入模型，这一步把因果推断模块装进了模型里
    model = TAPB(c=c, p_ci=p_ci, model_configs=model_configs).to(device)

    # 10. 设置优化器
    # 使用 AdamW 优化器，只更新 requires_grad=True 的参数
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    # 11. 加载 ESM-2 特征数据
    protein_f = open(protein_path, 'rb')
    pr_f = pickle.load(protein_f)  # 把之前提取的巨大的 ESM-2 特征列表读入内存

    # 12. 构建 Dataset 对象
    # 将 DataFrame 索引、原始数据、ESM-2特征 打包
    train_dataset = DTIDataset(df_train.index.values, df_train, pr_f)
    val_dataset = DTIDataset(df_val.index.values, df_val, pr_f)
    test_dataset = DTIDataset(df_test.index.values, df_test, pr_f)

    # 13. 构建 DataLoader (数据传送带)
    # 加载 Molformer 的分词器 (Tokenizer)
    drug_tokenizer = AutoTokenizer.from_pretrained(mol_path, trust_remote_code=True)
    bz = config.TRAIN.BATCH_SIZE
    MLM = config.TRAIN.MLM

    # 训练集加载器：开启了 Shuffle (打乱)，传入了 aa_dict (用于突变)，设置了 MLM、Mask率、删除率、突变率
    # 这些参数直接控制了 TAPB 的去偏训练策略
    train_dataloader = get_dataLoader(bz, train_dataset, drug_tokenizer, aa=aa_dict, shuffle=True, MLM=MLM,
                                      mask_rate=config.TRAIN.MASK_PROBABILITY,
                                      target_random_deletion_ratio=config.TRAIN.TARGET_RANDOM_DROP_RATIO,
                                      mutation_rate=config.TRAIN.MUTAION)

    # 验证/测试集加载器：干净的数据，没有任何 Mask 或突变
    val_dataloader = get_dataLoader(bz, val_dataset, drug_tokenizer)
    test_dataloader = get_dataLoader(bz, test_dataset, drug_tokenizer)

    # 14. 开始训练
    # 初始化训练器
    trainer = Trainer(model, opt, device, train_dataloader, val_dataloader, test_dataloader, output_path, config)
    # 启动训练循环！并返回最佳结果
    result, best_epoch = trainer.train()

    # 15. 保存模型架构信息
    with open(os.path.join(output_path, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    print()

    return result, best_epoch

if __name__ == '__main__':
    s = time()
    result, best_epoch = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
