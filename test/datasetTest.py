import pandas as pd
import networkx as nx
import numpy as np
import torch
# import



if __name__ == '__main__':
    data = pd.read_csv('D:\\学习\\论文\\2024new-第二篇\\pmy\\dataset\\NF-BoT-IoT\\data\\NF-BoT-IoT.csv')

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
    max_length = max(data_lengths)

    # # 在随机位置插入空值
    # for group in filtered_IP_list:
    #     data_row = group["data"]
    #     current_length = len(data_row)
    #
    #     if current_length < max_length:
    #         num_rows_to_add = max_length - current_length
    #
    #         # 构建空值 DataFrame，与原数据保持相同的列名
    #         zero_rows = pd.DataFrame(0, index=range(num_rows_to_add), columns=data_row.columns)
    #
    #         # # 随机插入空值
    #         # insertion_indices = np.random.choice(range(current_length + num_rows_to_add), size=num_rows_to_add,
    #         #                                      replace=False)
    #
    #         # 指定插入位置（这里以固定位置为例，可根据需求更改）
    #         insertion_indices = np.linspace(0, current_length, num=num_rows_to_add, endpoint=False, dtype=int)
    #
    #         # 将原数据和全 0 行合并
    #         for i, index in enumerate(insertion_indices):
    #             data_row = pd.concat([data_row.iloc[:index], zero_rows.iloc[:1], data_row.iloc[index:]]).reset_index(drop=True)
    #             zero_rows = zero_rows.iloc[1:]  # 更新 zero_rows 剩余部分
    #
    #         # 更新组的 data
    #         group["data"] = data_row

    # 替换数据中的部分行为全零行
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
            group["data"] = full_data

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    data = pd.read_csv("D:\\学习\\论文\\2024new-第二篇\\pmy\\DLGNN\\PreprocessedDataSet\\Preprocessed_NF-BoT-IoT.csv")
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
    max_length = max(data_lengths)

    # 在随机位置插入空值
    for group in filtered_IP_list:
        data = group["data"]
        current_length = len(data)

        if current_length < max_length:
            num_rows_to_add = max_length - current_length

            # 构建空值 DataFrame，与原数据保持相同的列名
            zero_rows = pd.DataFrame(0, index=range(num_rows_to_add), columns=data.columns)

            # # 随机插入空值
            # insertion_indices = np.random.choice(range(current_length + num_rows_to_add), size=num_rows_to_add,
            #                                      replace=False)

            # 指定插入位置（这里以固定位置为例，可根据需求更改）
            insertion_indices = np.linspace(0, current_length, num=num_rows_to_add, endpoint=False, dtype=int)

            # 将原数据和全 0 行合并
            for index in insertion_indices:
                data = pd.concat([data.iloc[:index], zero_rows.iloc[:1], data.iloc[index:]]).reset_index(drop=True)
                zero_rows = zero_rows.iloc[1:]  # 更新 zero_rows 剩余部分

            # 更新组的 data
            group["data"] = data



    FinalDataList=[]
    for i in max_length:
        combined_data = []
        # 遍历 filtered_data_list
        for group in filtered_IP_list:
            source_ip = group["source_ip"]
            dst_ip = group["dst_ip"]
            data = group["data"]

            # 从每组数据中提取一行（例如：第一行）
            extracted_row = data.iloc[i]  # 或 np.random.choice(len(data)) 选择随机行
            combined_data.append(extracted_row)

            # 添加到 最终的数据集
            FinalDataList.append({
                "source_ip": source_ip,
                "dst_ip": dst_ip,
                "data": extracted_row.to_frame().T  # 转换为 DataFrame 格式
            })

        # # 将提取的数据组合成一个新的 DataFrame
        # combined_df = pd.DataFrame(combined_data)

        # 转换为 DataFrame
        save_df = pd.DataFrame(FinalDataList)

        # 保存为 CSV 文件
        csv_path = "D:\\学习\\论文\\2024new-第二篇\\pmy\\DLGNN\\PreprocessedDataSet\\Preprocessed_NF-BoT-IoT.csv"
        save_df.to_csv(csv_path, index=False)
        print(f"数据已保存到 {csv_path}")








