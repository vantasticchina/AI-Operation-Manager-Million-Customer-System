"""
SHAP分析：XGBoost模型的全局和局部解释
用于分析客户未来3个月资产是否能提升至100万+的预测模型
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SHAPXGBoostAnalyzer:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.class_names = ['不会升级', '会升级']
        self.X_train = None
        self.X_test = None
        
    def prepare_features(self, df):
        """
        准备特征数据，创建目标变量
        """
        df_features = df.copy()
        
        # 定义目标：当前资产等级为'80-100万'的客户最有可能升级到'100万+'
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
        
        # App活跃度特征
        df_features['app_activity_score'] = (
            df_features['app_login_count'] * 0.3 + 
            df_features['app_financial_view_time'] * 0.4 + 
            df_features['app_product_compare_count'] * 0.3
        )
        
        # 投资活跃度特征
        df_features['investment_activity_score'] = (
            df_features['investment_monthly_count'] * 0.7 + 
            df_features['financial_repurchase_count'] * 0.3
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
    
    def train_model(self, df, params=None):
        """
        训练XGBoost模型
        """
        if params is None:
            # 设置XGBoost参数
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'random_state': 42,
                'eval_metric': 'auc'
            }
        
        X, y = self.prepare_features(df)
        
        # 输出目标变量分布
        print(f"目标变量分布 - 潜在升级客户 (1): {sum(y == 1)}, 非升级客户 (0): {sum(y == 0)}")
        print(f"升级客户占比: {sum(y == 1) / len(y) * 100:.2f}%")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 保存训练和测试数据
        self.X_train = X_train
        self.X_test = X_test
        
        # 创建XGBoost数据集
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # 训练模型
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        print("\nXGBoost模型训练完成")
        return X_train, X_test, y_train, y_test
    
    def global_shap_analysis(self, sample_size=1000):
        """
        进行全局SHAP分析
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        # 为SHAP分析选择一个样本（如果数据太大，选择子集）
        if len(self.X_train) > sample_size:
            X_sample = self.X_train.sample(n=sample_size, random_state=42)
        else:
            X_sample = self.X_train
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # 计算特征的平均绝对SHAP值（全局重要性）
        shap_importance = np.abs(shap_values).mean(0)
        shap_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Importance': shap_importance
        }).sort_values(by='SHAP_Importance', ascending=False)
        
        # 绘制全局重要性图
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(shap_importance_df.head(15))), 
                shap_importance_df.head(15)['SHAP_Importance'], 
                alpha=0.7)
        plt.yticks(range(len(shap_importance_df.head(15))), 
                  shap_importance_df.head(15)['Feature'])
        plt.xlabel('平均SHAP值 (特征重要性)')
        plt.title('SHAP全局特征重要性 (前15位)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('D:/博客内容/day02/prediction_models/xgboost/shap_global_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 绘制SHAP摘要图
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('SHAP摘要图 - 全局特征影响')
        plt.tight_layout()
        plt.savefig('D:/博客内容/day02/prediction_models/xgboost/shap_summary_plot.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("全局SHAP分析结果 (前10位特征):")
        print(shap_importance_df.head(10))
        
        return shap_values, shap_importance_df, X_sample
    
    def local_shap_analysis(self, shap_values, X_sample, indices=None):
        """
        进行局部SHAP分析，解释单个客户的预测
        """
        if indices is None:
            # 选择几个有代表性的客户进行分析
            indices = [0, 1, 2, 3, 4]  # 分析前5个客户
        
        explainer = shap.TreeExplainer(self.model)
        
        # 对每个客户生成SHAP力图
        for idx in indices:
            if idx >= len(X_sample):
                continue
                
            print(f"\n局部SHAP分析 - 客户 {idx}:")
            print(f"特征值: {X_sample.iloc[idx].values}")
            
            # 获取该客户的SHAP值
            shap_vals_single = shap_values[idx]
            
            # 使用SHAP力图展示单个预测的解释
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals_single,
                    base_values=explainer.expected_value,
                    data=X_sample.iloc[idx],
                    feature_names=self.feature_names
                ), 
                max_display=10,
                show=False
            )
            plt.title(f'SHAP力图 - 客户 {idx} 的预测解释')
            plt.tight_layout()
            plt.savefig(f'D:/博客内容/day02/prediction_models/xgboost/shap_local_client_{idx}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # 打印该客户的主要影响特征
            feature_importance = list(zip(self.feature_names, shap_vals_single))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"客户 {idx} 的主要影响特征 (前5位):")
            for i, (feature, shap_val) in enumerate(feature_importance[:5]):
                impact = "正向" if shap_val > 0 else "负向"
                print(f"  {i+1}. {feature}: {shap_val:.4f} ({impact}影响)")
    
    def analyze_predictions(self):
        """
        分析预测结果，结合SHAP值解释高价值客户特征
        """
        # 使用测试数据进行预测
        dtest = xgb.DMatrix(self.X_test)
        predictions = self.model.predict(dtest)
        
        # 分析预测概率较高的客户
        high_prob_indices = np.where(predictions > 0.7)[0][:10]  # 前10个高概率客户
        low_prob_indices = np.where(predictions < 0.3)[0][:10]   # 前10个低概率客户
        
        print(f"高概率客户 (预测概率 > 0.7) 数量: {len(high_prob_indices)}")
        print(f"低概率客户 (预测概率 < 0.3) 数量: {len(low_prob_indices)}")
        
        # 为高概率和低概率客户生成SHAP解释
        explainer = shap.TreeExplainer(self.model)
        
        # 获取高概率客户的SHAP值
        if len(high_prob_indices) > 0:
            X_high_prob = self.X_test.iloc[high_prob_indices]
            shap_values_high = explainer.shap_values(X_high_prob)
            
            print(f"\n高概率客户 (前3个) 的平均SHAP值:")
            avg_shap_high = np.mean(shap_values_high, axis=0)
            high_features = list(zip(self.feature_names, avg_shap_high))
            high_features.sort(key=lambda x: abs(x[1]), reverse=True)
            for i, (feature, shap_val) in enumerate(high_features[:5]):
                impact = "正向" if shap_val > 0 else "负向"
                print(f"  {feature}: {shap_val:.4f} ({impact}影响)")
        
        # 获取低概率客户的SHAP值
        if len(low_prob_indices) > 0:
            X_low_prob = self.X_test.iloc[low_prob_indices]
            shap_values_low = explainer.shap_values(X_low_prob)
            
            print(f"\n低概率客户 (前3个) 的平均SHAP值:")
            avg_shap_low = np.mean(shap_values_low, axis=0)
            low_features = list(zip(self.feature_names, avg_shap_low))
            low_features.sort(key=lambda x: abs(x[1]), reverse=True)
            for i, (feature, shap_val) in enumerate(low_features[:5]):
                impact = "正向" if shap_val > 0 else "负向"
                print(f"  {feature}: {shap_val:.4f} ({impact}影响)")


def main():
    # 读取数据
    print("=== SHAP分析：XGBoost模型的全局和局部解释 ===")
    print("正在加载数据...")
    customer_base_df = pd.read_csv('D:/博客内容/day02/customer_base.csv')
    customer_behavior_df = pd.read_csv('D:/博客内容/day02/customer_behavior_assets.csv')
    
    # 合并数据
    merged_df = pd.merge(customer_base_df, customer_behavior_df, on='customer_id', how='inner')
    
    # 过滤掉资产已经是100万+的客户（他们无法再升级）
    df_filtered = merged_df[merged_df['asset_level'] != '100万+'].copy()
    
    print(f"\n数据集信息:")
    print(f"总客户数: {len(df_filtered)}")
    print(f"资产等级分布:\n{df_filtered['asset_level'].value_counts()}")
    
    # 创建分析器实例
    analyzer = SHAPXGBoostAnalyzer()
    
    # 训练模型
    print("\n" + "="*50)
    print("正在训练XGBoost模型...")
    X_train, X_test, y_train, y_test = analyzer.train_model(df_filtered)
    
    # 全局SHAP分析
    print("\n" + "="*50)
    print("正在进行全局SHAP分析...")
    shap_values, shap_importance_df, X_sample = analyzer.global_shap_analysis()
    
    # 局部SHAP分析
    print("\n" + "="*50)
    print("正在进行局部SHAP分析...")
    analyzer.local_shap_analysis(shap_values, X_sample, indices=[0, 1, 2, 100, 1000])
    
    # 预测分析
    print("\n" + "="*50)
    print("正在分析预测结果...")
    analyzer.analyze_predictions()
    
    # 总结
    print("\n" + "="*50)
    print("SHAP分析总结:")
    print("- 全局解释: 识别了对预测最重要的特征")
    print("- 局部解释: 解释了单个客户的预测原因")
    print("- 业务洞察: 了解哪些因素驱动高价值客户预测")


if __name__ == "__main__":
    main()