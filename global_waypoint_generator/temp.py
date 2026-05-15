import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

def visualize_feature_distributions(npz_file_path, save_dir='./output_plots'):
    print(f"正在加载数据: {npz_file_path}")
    data = np.load(npz_file_path)
    points = data['points']  # shape: [N, 9]
    labels = data['labels']  # shape: [N, 2]

    feature_names = ['f_start', 'f_goal', 'd_obs_norm', 'nx', 'ny']
    
    # 1. 提取有效数据
    features = points[2:, 3:8]
    point_scores = labels[2:, 0] 

    # 2. 划分正负样本
    wp_mask = point_scores > 0.8
    bg_mask = point_scores < 0.1

    wp_features = features[wp_mask]
    bg_features = features[bg_mask]

    print(f"提取到正样本(航路点附近): {len(wp_features)} 个")
    print(f"提取到负样本(背景点): {len(bg_features)} 个")

    if len(wp_features) == 0:
        print("该样本中没有得分 > 0.8 的非起终点数据，请检查标签生成逻辑或降低阈值。")
        return

    # 3. 构建 DataFrame
    df_wp = pd.DataFrame(wp_features, columns=feature_names)
    df_wp['Class'] = 'Waypoint (Score > 0.8)'

    df_bg = pd.DataFrame(bg_features, columns=feature_names)
    df_bg['Class'] = 'Background (Score < 0.1)'

    df_all = pd.concat([df_wp, df_bg], ignore_index=True)

    # 4. 绘制特征的一维概率密度分布
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        sns.histplot(
            data=df_all, 
            x=feat,               
            hue='Class',          
            stat="density",       
            common_norm=False,    
            kde=True,             
            alpha=0.4,            
            ax=axes[i],
            palette=['#FF5722', '#03A9F4'] 
        )
        
        axes[i].set_title(f'Distribution of {feat}')
        axes[i].set_xlabel("Feature Value")
        axes[i].set_ylabel("Density")

    plt.suptitle("Feature Distribution Analysis", fontsize=16)
    plt.tight_layout()

    # ==========================================
    # 💡 修改点 2：保存图片逻辑 (必须在 plt.show 之前)
    # ==========================================
    # 确保保存的文件夹存在，如果不存在则自动创建
    os.makedirs(save_dir, exist_ok=True)
    
    # 构造唯一的文件名（这里提取原始 npz 的名字，加上后缀）
    base_name = os.path.basename(npz_file_path).replace('.npz', '')
    save_path = os.path.join(save_dir, f"{base_name}_distribution.png")
    
    # 保存图片：dpi=300 保证清晰度，bbox_inches='tight' 防止边缘标题被裁切
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已成功保存至: {save_path}")

    # 如果你在服务器上跑，不需要弹窗，可以直接把下面这行注释掉
    plt.show()

    

    # # 5. [进阶] 绘制联合分布图 (Pairplot)
    # # 观察特征组合起来是否能更好地区分
    # print("正在绘制特征散点相关性矩阵...")
    # sns.pairplot(
    #     df_all, 
    #     vars=['nx', 'ny', 'd_obs_norm'], # 选几个主要标量特征
    #     hue='Class', 
    #     plot_kws={'alpha': 0.6, 's': 15},
    #     palette=['#FF5722', '#03A9F4']
    # )
    # plt.suptitle("Feature Pairplot (Joint Distributions)", y=1.02)
    # plt.show()

# 调用测试
if __name__ == "__main__":
    # 基础路径配置
    base_dir = r"C:\Users\Administrator\Desktop\experiments\train_data15"
    save_directory = os.path.join(base_dir, "plots")  # 保存图片的文件夹路径
    
    # 你想随机抽取测试的文件数量
    num_test_samples = 5 
    
    print(f"🚀 开始批量随机测试，计划抽取 {num_test_samples} 个文件...")
    
    for i in range(num_test_samples):
        # 💡 随机生成 map 和 task 的数字
        map_id = random.randint(0, 49)     # 生成 0 到 9 之间的随机整数
        task_id = random.randint(0, 9)   # 生成 0 到 49 之间的随机整数
        
        # 拼接出完整的文件名和路径
        file_name = f"map{map_id}_task{task_id}.npz"
        sample_file = os.path.join(base_dir, file_name)
        
        print("-" * 40)
        print(f"[{i+1}/{num_test_samples}] 尝试处理: {file_name}")
        
        if os.path.exists(sample_file):
            visualize_feature_distributions(sample_file, save_directory)
        else:
            print(f"❌ 未找到文件: {sample_file}，已跳过。")
            
    print("-" * 40)
    print("✅ 批量测试结束！请前往 plots 文件夹查看生成的图片。")


    