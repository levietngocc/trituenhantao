# File: src/preprocess.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
import logging

class Preprocessor:
    def __init__(self):
        self.logger = logging.getLogger('PipelineLogger')
        self.scaler = None
        self.pca = None
        
    def load_and_clean(self, raw_df):
        self.logger.info("Bắt đầu làm sạch dữ liệu...")
        df = raw_df.copy()
        
        # 1. Bỏ missing CustomerID
        df.dropna(subset=['CustomerID'], inplace=True)
        
        # 2. Lọc Quantity và UnitPrice > 0 (Tự động loại bỏ đơn hủy bắt đầu bằng 'C')
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        
        # 3. Chuyển kiểu dữ liệu
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
        
        # 4. Chỉ giữ lại United Kingdom để giảm nhiễu (như yêu cầu)
        df = df[df['Country'] == 'United Kingdom']
        
        self.logger.info(f"Làm sạch xong. Kích thước còn lại: {df.shape}")
        return df

    def build_rfm_features(self, cleaned_df):
        self.logger.info("Bắt đầu trích xuất đặc trưng RFM+...")
        cleaned_df['TotalPrice'] = cleaned_df['Quantity'] * cleaned_df['UnitPrice']
        max_date = cleaned_df['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        # Sử dụng groupby aggregation hiệu quả
        rfm = cleaned_df.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda x: (max_date - x.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalPrice', 'sum'),
            UniqueProducts=('StockCode', 'nunique'),
            TotalQuantity=('Quantity', 'sum')
        )
        
        rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
        rfm['AvgQuantityPerOrder'] = rfm['TotalQuantity'] / rfm['Frequency']
        
        self.logger.info(f"Tạo xong tập RFM cho {rfm.shape[0]} khách hàng.")
        return rfm

    def handle_outliers(self, df, columns, method='iqr', factor=1.5):
        """Khuyến nghị: Không cắt outlier cho bài toán này, giữ lại khách VIP."""
        self.logger.info("Bỏ qua việc cắt outlier (giữ khách VIP).")
        return df

    def transform_features(self, df, log_cols=['Monetary', 'Frequency', 'AvgOrderValue', 'TotalQuantity', 'Recency', 'UniqueProducts', 'AvgQuantityPerOrder']):
        self.logger.info("Áp dụng log1p transform...")
        df_transformed = df.copy()
        for col in log_cols:
            # np.log1p an toàn cho giá trị 0
            df_transformed[col] = np.log1p(df_transformed[col])
        return df_transformed

    def scale_features(self, df, scaler_type='robust'):
        self.logger.info(f"Scaling dữ liệu bằng {scaler_type}...")
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        X_scaled = self.scaler.fit_transform(df)
        return X_scaled

    def apply_pca(self, X, n_components=2):
        self.logger.info(f"Áp dụng PCA với {n_components} components...")
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        variance_ratio = sum(self.pca.explained_variance_ratio_)
        self.logger.info(f"Tổng phương sai giải thích: {variance_ratio:.4f}")
        return X_pca

    def get_processed_data(self, raw_df):
        cleaned_df = self.load_and_clean(raw_df)
        rfm_raw = self.build_rfm_features(cleaned_df)
        
        # Tiền xử lý
        rfm_log = self.transform_features(rfm_raw)
        X_scaled = self.scale_features(rfm_log, scaler_type='robust')
        X_pca = self.apply_pca(X_scaled, n_components=2)
        
        return {
            'X_scaled': X_scaled,
            'X_pca': X_pca,
            'feature_names': rfm_raw.columns.tolist(),
            'customer_ids': rfm_raw.index.values,
            'rfm_raw': rfm_raw,
            'scaler': self.scaler,
            'pca': self.pca
        }