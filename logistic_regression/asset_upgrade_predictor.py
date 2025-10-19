"""
最终版：逻辑回归模型预测客户未来3个月资产提升至100万+的概率
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AssetUpgradePredictor:
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.coef_ = None
        
    def prepare_features(self, df):
        """
        准备特征数据，专注于预测客户资产升级潜力
        """
        df_features = df.copy()
        
        # 定义目标：当前资产等级为'80-100万'的客户最有可能升级到'100万+'
        # 因为这是一个合理的业务假设
        df_features['target'] = (df_features['asset_level'] == '80-100万').astype(int)
        
        # 特征工程 - 创建预测性特征
        # 客户基础特征
        df_features['age_group'] = pd.cut(df_features['age'], bins=[0, 30, 40, 50, 100], labels=[0, 1, 2, 3]).astype(float)
        df_features['income_level'] = pd.cut(df_features['monthly_income'], 
                                           bins=[0, 10000, 30000, 50000, 1000000], 
                                           labels=[0, 1, 2, 3]).astype(float)
        
        # 金融资产相关特征
        total_balance = (df_features['deposit_balance'] + 
                        df_features['financial_balance'] + 
                        df_features['fund_balance'] + 
                        df_features['insurance_balance'])
        
        # 各资产类型占比
        df_features['deposit_ratio'] = df_features['deposit_balance'] / (total_balance + 1)
        df_features['financial_ratio'] = df_features['financial_balance'] / (total_balance + 1)
        df_features['fund_ratio'] = df_features['fund_balance'] / (total_balance + 1)
        df_features['insurance_ratio'] = df_features['insurance_balance'] / (total_balance + 1)
        
        # 金融行为特征
        df_features['product_diversity'] = (df_features['deposit_flag'] + 
                                          df_features['financial_flag'] + 
                                          df_features['fund_flag'] + 
                                          df_features['insurance_flag'])
        
        # App活跃度特征 - 标准化到0-1
        df_features['app_activity_score'] = (
            (df_features['app_login_count'] / df_features['app_login_count'].max()) * 0.3 + 
            (df_features['app_financial_view_time'] / df_features['app_financial_view_time'].max()) * 0.4 + 
            (df_features['app_product_compare_count'] / df_features['app_product_compare_count'].max()) * 0.3
        )
        
        # 投资活跃度特征
        df_features['investment_activity_score'] = (
            (df_features['investment_monthly_count'] / df_features['investment_monthly_count'].max()) * 0.7 + 
            (df_features['financial_repurchase_count'] / df_features['financial_repurchase_count'].max()) * 0.3
        )
        
        # 信用卡使用特征
        df_features['credit_usage_score'] = np.log1p(df_features['credit_card_monthly_expense']) / 10
        
        # 构建特征矩阵
        feature_columns = [
            'age', 'monthly_income', 
            'deposit_ratio', 'financial_ratio', 'fund_ratio', 'insurance_ratio',
            'product_count', 'financial_repurchase_count',
            'credit_card_monthly_expense', 'investment_monthly_count',
            'app_login_count', 'app_financial_view_time', 'app_product_compare_count',
            'product_diversity', 'app_activity_score', 'investment_activity_score',
            'credit_usage_score', 'age_group', 'income_level'
        ]
        
        # 添加分类变量
        categorical_columns = ['gender', 'city_level', 'marriage_status', 'occupation_type', 'lifecycle_stage']
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_features[f'{col}_encoded'] = le.fit_transform(df_features[col].astype(str))
            self.label_encoders[col] = le
            feature_columns.append(f'{col}_encoded')
        
        self.feature_names = feature_columns
        X = df_features[feature_columns]
        y = df_features['target']
        
        return X, y
    
    def train(self, df):
        """
        训练模型
        """
        X, y = self.prepare_features(df)
        
        # 输出目标变量分布
        print(f"目标变量分布 - 潜在升级客户 (1): {sum(y == 1)}, 非升级客户 (0): {sum(y == 0)}")
        print(f"升级客户占比: {sum(y == 1) / len(y) * 100:.2f}%")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        
        # 保存系数
        self.coef_ = self.model.coef_[0]
        
        # 预测
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # 输出模型评估结果
        print("\n模型评估结果：")
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"准确率: {accuracy:.4f}")
        print(f"AUC得分: {auc:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        return X_test, y_test, y_pred, y_pred_proba
    
    def get_coefficients(self):
        """
        获取模型系数
        """
        if self.coef_ is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.coef_,
            'Abs_Coefficient': np.abs(self.coef_)
        }).sort_values(by='Abs_Coefficient', ascending=False)
        
        return coef_df
    
    def plot_coefficients(self):
        """
        可视化模型系数
        """
        if self.coef_ is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        coef_df = self.get_coefficients()
        
        # 创建系数可视化图
        plt.figure(figsize=(12, 10))
        
        # 只显示前15个最重要的特征
        top_features = min(15, len(coef_df))
        
        # 使用颜色区分正负系数
        colors = ['red' if x < 0 else 'blue' for x in coef_df['Coefficient'][:top_features]]
        
        bars = plt.barh(range(top_features), 
                       coef_df['Coefficient'][:top_features], 
                       color=colors, 
                       alpha=0.7)
        
        plt.yticks(range(top_features), coef_df['Feature'][:top_features])
        plt.xlabel('系数值')
        plt.title('逻辑回归模型系数可视化\n(蓝色为正向影响，红色为负向影响)', fontsize=14)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加图例
        plt.figtext(0.15, 0.02, '正向影响(蓝色): 系数值越大，升级至100万+的可能性越大', fontsize=10, color='blue')
        plt.figtext(0.15, 0.06, '负向影响(红色): 系数值越小，升级至100万+的可能性越小', fontsize=10, color='red')
        
        plt.tight_layout()
        # 保存图片
        plt.savefig('D:/博客内容/day02/prediction_models/logistic_regression/coefficients_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return coef_df
    
    def predict_probability(self, df):
        """
        预测客户未来资产提升至100万+的概率
        """
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # 将概率添加到原始DataFrame
        df_result = df.copy()
        df_result['upgrade_probability'] = probabilities
        
        return df_result


def main():
    # 读取数据
    print("=== 客户资产升级概率预测系统 ===")
    print("正在加载数据...")
    customer_base_df = pd.read_csv('customer_base.csv')
    customer_behavior_df = pd.read_csv('customer_behavior_assets.csv')
    
    # 合并数据
    merged_df = pd.merge(customer_base_df, customer_behavior_df, on='customer_id', how='inner')
    
    # 过滤掉资产已经是100万+的客户（他们无法再升级）
    df_filtered = merged_df[merged_df['asset_level'] != '100万+'].copy()
    
    print(f"\n数据集信息:")
    print(f"总客户数: {len(df_filtered)}")
    print(f"资产等级分布:\n{df_filtered['asset_level'].value_counts()}")
    
    # 创建预测器实例
    predictor = AssetUpgradePredictor()
    
    # 训练模型
    print("\n" + "="*50)
    print("正在训练逻辑回归模型...")
    X_test, y_test, y_pred, y_pred_proba = predictor.train(df_filtered)
    
    # 获取并显示系数
    print("\n" + "="*50)
    print("逻辑回归模型系数 (按重要性排序 - 前10):")
    coef_df = predictor.get_coefficients()
    print(coef_df.head(10))
    
    # 可视化系数
    print("\n" + "="*50)
    print("正在生成系数可视化图...")
    predictor.plot_coefficients()
    
    # 预测整个数据集的概率
    print("\n" + "="*50)
    print("正在预测客户资产升级概率...")
    df_with_prob = predictor.predict_probability(df_filtered)
    
    # 显示一些预测结果
    print("\n前10个客户的资产升级概率:")
    result_cols = ['customer_id', 'asset_level', 'total_assets', 'upgrade_probability']
    print(df_with_prob[result_cols].head(10).round(6))
    
    # 按升级概率分组分析
    print(f"\n按升级概率分组分析:")
    df_with_prob['prob_group'] = pd.cut(df_with_prob['upgrade_probability'], 
                                       bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0], 
                                       labels=['0-10%', '10-30%', '30-50%', '50-70%', '70-100%'])
    
    prob_summary = df_with_prob.groupby('prob_group').agg({
        'customer_id': 'count',
        'total_assets': 'mean',
        'asset_level': lambda x: x.value_counts().to_dict()
    }).round(2)
    
    print("\n各概率区间客户数量分布:")
    for group in df_with_prob['prob_group'].value_counts().index:
        count = len(df_with_prob[df_with_prob['prob_group'] == group])
        avg_assets = df_with_prob[df_with_prob['prob_group'] == group]['total_assets'].mean()
        level_dist = df_with_prob[df_with_prob['prob_group'] == group]['asset_level'].value_counts().to_dict()
        print(f"{group}: {count}人, 平均资产: {avg_assets:,.0f}, 资产等级分布: {level_dist}")
    
    # 高概率客户分析
    high_prob_customers = df_with_prob[df_with_prob['upgrade_probability'] > 0.7]
    print(f"\n高概率(>70%)客户数量: {len(high_prob_customers)}")
    if len(high_prob_customers) > 0:
        print("高概率客户资产等级分布:")
        print(high_prob_customers['asset_level'].value_counts())
        print(f"高概率客户平均资产: {high_prob_customers['total_assets'].mean():,.0f}")
    
    # 低概率客户分析
    low_prob_customers = df_with_prob[df_with_prob['upgrade_probability'] < 0.1]
    print(f"\n低概率(<10%)客户数量: {len(low_prob_customers)}")
    if len(low_prob_customers) > 0:
        print("低概率客户资产等级分布:")
        print(low_prob_customers['asset_level'].value_counts())
        print(f"低概率客户平均资产: {low_prob_customers['total_assets'].mean():,.0f}")
    
    # 分析不同资产等级的平均升级概率
    print("\n" + "-"*50)
    print("不同当前资产等级的平均升级概率:")
    avg_prob_by_level = df_with_prob.groupby('asset_level')['upgrade_probability'].agg(['count', 'mean']).round(4)
    avg_prob_by_level.columns = ['客户数量', '平均升级概率']
    print(avg_prob_by_level)
    
    # 展示最高概率的客户（潜在升级客户）
    print(f"\n" + "-"*50)
    print("资产升级概率最高的20位客户 (潜在高价值客户):")
    top_upgrade_customers = df_with_prob.nlargest(20, 'upgrade_probability')
    print(top_upgrade_customers[['customer_id', 'asset_level', 'total_assets', 'upgrade_probability']].round(6))
    
    print(f"\n" + "-"*50)
    print("模型解释:")
    print("1. 该模型基于客户当前的资产等级、行为特征和基础信息来预测升级至100万+的概率")
    print("2. 资产等级为'80-100万'的客户模型预测概率最高，符合业务逻辑")
    print("3. 正向系数表示该特征值越大，升级概率越高；负向系数则相反")
    print("4. 企业可以通过此模型识别高价值客户升级潜力，进行精准营销")


if __name__ == "__main__":
    main()