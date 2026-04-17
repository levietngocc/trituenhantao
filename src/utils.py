# File: src/utils.py
import os
import random
import numpy as np
import pandas as pd
import logging

def set_seed(seed=42):
    """Cố định seed cho các thư viện để kết quả có thể tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Nếu dùng scikit-learn, các hàm sẽ nhận tham số random_state trực tiếp.

def ensure_dir(directory):
    """Tạo thư mục nếu chưa tồn tại."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_logger(name, log_file=None):
    """Thiết lập logger ghi ra console và file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Tránh duplicate log nếu gọi nhiều lần
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        if log_file:
            ensure_dir(os.path.dirname(log_file))
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
    return logger

def load_data(filepath):
    """Đọc dữ liệu từ file CSV hoặc Excel."""
    logger = logging.getLogger('PipelineLogger')
    if not os.path.exists(filepath):
        logger.error(f"File không tồn tại: {filepath}")
        raise FileNotFoundError(f"Vui lòng tải dữ liệu và đặt vào: {filepath}")
    
    logger.info(f"Đang tải dữ liệu từ {filepath}...")
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, encoding='latin1')
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Định dạng file không được hỗ trợ (chỉ csv/xlsx).")
        logger.info(f"Tải thành công. Kích thước dữ liệu: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Lỗi khi đọc file: {e}")
        raise

def save_dataframe(df, path, index=False):
    """Lưu DataFrame ra file CSV."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)
    logging.getLogger('PipelineLogger').info(f"Đã lưu dữ liệu vào {path}")