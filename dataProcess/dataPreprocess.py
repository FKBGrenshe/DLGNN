import pandas as pd
import networkx as nx
import numpy as np
import torch
# from GraphBuild import *
from dataProcess.GraphBuild import *
from sklearn.preprocessing import StandardScaler
def Step1(max_length):

    data = pd.read_csv('D:\\学习\\论文\\2024new-第二篇\\pmy\\dataset\\NF-BoT-IoT\\data\\NF-BoT-IoT.csv')
    # filtered_IP_list = pd.read_csv('D:\\学习\\论文\\2024new-第二篇\\pmy\\DLGNN\\PreprocessedDataSet\\Preprocessed_NF-BoT-IoT.csv')

    '新添加特征'
    data['BYTESPERPACKET'] = (data["IN_BYTES"] + data["OUT_BYTES"]) / (data["IN_PKTS"] + data["OUT_PKTS"])
    '''
    数据标准化
    '''
    # 数值标准化（零均值和单位方差）
    scaler = StandardScaler()
    data['PROTOCOL'] = scaler.fit_transform(data[['PROTOCOL']])
    data['FLOW_DURATION_MILLISECONDS'] = scaler.fit_transform(data[['FLOW_DURATION_MILLISECONDS']])
    data['IN_BYTES'] = scaler.fit_transform(data[['IN_BYTES']])
    data['BYTESPERPACKET'] = scaler.fit_transform(data[['BYTESPERPACKET']])
    data['TCP_FLAGS'] = scaler.fit_transform(data[['TCP_FLAGS']])

    filtered_IP_list = []
    # 提取 sourceIP 和 dstIP 的所有唯一组合
    ip_combinations = data[['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']].drop_duplicates()
    # 遍历每个组合
    for _, row in ip_combinations.iterrows():
        source_ip = row['IPV4_SRC_ADDR']
        dst_ip = row['IPV4_DST_ADDR']
        # 筛选出匹配的行
        filtered_data = data[(data['IPV4_SRC_ADDR'] == source_ip) & (data['IPV4_DST_ADDR'] == dst_ip)]
        # 将筛选结果存入数组（包含元信息）
        filtered_IP_list.append({
            "IPV4_SRC_ADDR": source_ip,
            "IPV4_DST_ADDR": dst_ip,
            "data": filtered_data
        })

    # 找出每个 IP 组的 data 个数
    data_lengths = [len(group["data"]) for group in filtered_IP_list]


    if (max_length == 0): # 没有提前说明最大长度
        max_length = max(data_lengths)



    # 在随机位置插入空值
    for group in filtered_IP_list:
        data = group["data"]
        current_length = len(data)

        # 如果当前数据长度小于 max_length，补齐为 max_length 行（全零行替换）
        if current_length < max_length:
            # 随机生成替换索引
            # replace_indices = np.random.choice(range(max_length), size=current_length, replace=False)
            replace_indices = np.sort(np.random.choice(range(max_length), size=current_length, replace=False))

            # 构建全零行 DataFrame
            zero_rows = pd.DataFrame(0, index=range(max_length), columns=data.columns)

            # 创建新的 DataFrame，将全零行替换到指定位置
            full_data = zero_rows.copy()
            # for idx, row in enumerate(data.itertuples(index=False)):
            #     full_data.iloc[replace_indices[idx]] = row
            for idx, row in enumerate(data.itertuples(index=False)):
                full_data.iloc[replace_indices[idx]] = pd.DataFrame([row])  # 仅保留数值部分
                # 更新 group 数据
                group["data"] = full_data.head(max_length)
        else:
            # 更新 group 数据
            group["data"] = data.head(max_length)

    saving = False
    if saving :
        # 转换为 DataFrame
        save_df = pd.DataFrame(filtered_IP_list)
        # 保存为 CSV 文件
        csv_path = "//root//autodl-tmp//DLGNN//PreprocessedDataSet//Preprocessed_NF-BoT-IoT.csv"
        save_df.to_csv(csv_path, index=False)
        print(f"数据已保存到 {csv_path}")

    FinalDataList = []
    for i in range(max_length):
        combined_data = []
        # 遍历 filtered_data_list
        for group in filtered_IP_list:
            source_ip = group["IPV4_SRC_ADDR"]
            dst_ip = group["IPV4_DST_ADDR"]
            data = group["data"]

            # 从每组数据中提取一行（例如：第一行）
            extracted_row = data.iloc[i]  # 或 np.random.choice(len(data)) 选择随机行
            combined_data.append(extracted_row)

            # 添加到 最终的数据集
            FinalDataList.append({
                "IPV4_SRC_ADDR": source_ip,
                "IPV4_DST_ADDR": dst_ip,
                "data": extracted_row.to_frame().T  # 转换为 DataFrame 格式
            })

        # # 将提取的数据组合成一个新的 DataFrame
        # combined_df = pd.DataFrame(combined_data)

        if saving:
            # 转换为 DataFrame
            save_df = pd.DataFrame(FinalDataList)

            # 保存为 CSV 文件
            csv_path = "//root//autodl-tmp//DLGNN//PreprocessedDataSet//Processed_NF-BoT-IoT.csv"
            save_df.to_csv(csv_path, index=False)
            print(f"数据已保存到 {csv_path}")




    return FinalDataList, [ item["data"]["Label"].iloc[0] for item in FinalDataList ]


if __name__ == '__main__':
    ## NF-BoT-IoT.csv总共181个源IP目的IP组合，因此总共181条边
    totalData, totalLable = Step1(10)
    snapshots = build_graph(totalData,181)
    # '生成线图快照'
    # line_graph_snapshots = generate_line_graph_snapshots(snapshots)
    # '生成线图邻接矩阵和特征矩阵'
    # adjacency_matrices, feature_matrices = generate_line_graph_matrices(line_graph_snapshots)
    # '生成线图邻接矩阵和特征矩阵'
    adjacency_matrices, feature_matrices = generate_line_graph_matrices(snapshots)