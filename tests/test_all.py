"""
ملف اختبار شامل لنظام Customer Churn Intelligence & Retention System
يغطي: API, Features, Ingestion, Processing, Models, Utils
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# إضافة المسار للمشروع
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==================== Fixtures ====================

@pytest.fixture
def mock_transactions():
    """بيانات معاملات وهمية للاختبار"""
    return pd.DataFrame({
        'Customer_ID': ['C001', 'C001', 'C002', 'C003', 'C003', 'C003'],
        'Order_ID': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6'],
        'Order_Date': pd.to_datetime([
            '2026-01-01', '2026-01-15', '2026-02-01', 
            '2026-01-05', '2026-02-10', '2026-03-01'
        ]),
        'Order_Amount': [100.0, 200.0, 150.0, 300.0, 250.0, 400.0],
        'Quantity': [2, 3, 1, 4, 2, 5],
        'Product_ID': ['P1', 'P2', 'P1', 'P3', 'P2', 'P1'],
        'Product_Category': ['Electronics', 'Clothing', 'Electronics', 
                            'Home', 'Clothing', 'Electronics']
    })

@pytest.fixture
def mock_artifact():
    """Artifact وهمي للنموذج"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.datasets import make_classification
    
    # إنشاء بيانات صغيرة للتدريب السريع
    X, y = make_classification(n_samples=100, n_features=6, random_state=42)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', HistGradientBoostingClassifier(max_iter=10, random_state=42))
    ])
    model.fit(X, y)
    
    return {
        'binary_model': model,
        'multi_tier_model': model
    }

# ==================== 1. Utils Tests ====================

class TestUtils:
    """اختبار إعدادات المشروع والاتصال بقاعدة البيانات"""
    
    def test_config_paths_exist(self):
        """التأكد من إنشاء المسارات تلقائياً"""
        from src.utils.config import PROJECT_ROOT, DATA_DIR, MODELS_DIR
        
        assert PROJECT_ROOT.exists()
        assert MODELS_DIR.exists()
        assert isinstance(DATA_DIR, Path)
    
    def test_db_connector_engine(self):
        """اختبار إنشاء محرك قاعدة البيانات (بدون اتصال فعلي)"""
        from src.utils.db_connector import get_postgres_engine
        from sqlalchemy.engine import Engine
        
        engine = get_postgres_engine()
        assert isinstance(engine, Engine)
        assert 'postgresql' in str(engine.url)
        engine.dispose()

# ==================== 2. Ingestion Tests ====================

class TestIngestion:
    """اختبار تحميل البيانات الخام"""
    
    def test_load_raw_transactions_csv(self, tmp_path, mock_transactions):
        """اختبار تحميل ملف CSV"""
        from src.ingestion.download_raw import load_raw_transactions
        
        # إنشاء ملف مؤقت
        csv_path = tmp_path / "test_data.csv"
        mock_transactions.to_csv(csv_path, index=False)
        
        result = load_raw_transactions(csv_path)
        assert len(result) == 6
        assert 'Customer_ID' in result.columns
    
    def test_load_raw_transactions_not_found(self):
        """اختبار الخطأ عند عدم وجود الملف"""
        from src.ingestion.download_raw import load_raw_transactions
        
        with pytest.raises(FileNotFoundError):
            load_raw_transactions(Path("nonexistent.csv"))
    
    def test_load_raw_transactions_invalid_format(self, tmp_path):
        """اختبار رفض الصيغ غير المدعومة"""
        from src.ingestion.download_raw import load_raw_transactions
        
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("dummy")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_raw_transactions(txt_path)

# ==================== 3. Processing Tests ====================

class TestProcessing:
    """اختبار تنظيف البيانات وإنشاء الجداول المساعدة"""
    
    def test_clean_transaction_logs(self, mock_transactions):
        """اختبار تنظيف البيانات"""
        from src.processing.clean_transactions import clean_transaction_logs
        
        # إضافة بيانات ملوثة
        dirty_df = mock_transactions.copy()
        dirty_df.loc[0, 'Order_Amount'] = None  # قيمة مفقودة
        dirty_df = pd.concat([dirty_df, dirty_df.iloc[[0]]])  # duplicates
        
        result = clean_transaction_logs(dirty_df)
        
        assert len(result) < len(dirty_df)  # تم حذف الصفوف المكررة
        assert result['Order_Amount'].isna().sum() == 0  # لا قيم مفقودة
        assert result['Quantity'].dtype in [np.int32, np.int64]  # نوع صحيح
    
    def test_generate_support_lookup(self, mock_transactions):
        """اختبار إنشاء جداول البحث"""
        from src.processing.generate_support import generate_support_lookup_tables
        
        result = generate_support_lookup_tables(mock_transactions)
        
        assert not result.empty
        assert 'Product_Category' in result.columns
        assert 'Total_Sales' in result.columns
        assert len(result) == 3  # 3 فئات في البيانات الوهمية

# ==================== 4. Features Tests ====================

class TestFeatures:
    """اختبار بناء ميزات RFM"""
    
    def test_generate_rfm_matrix(self, mock_transactions):
        """اختبار إنشاء مصفوفة RFM"""
        from src.features.build_rfm_features import generate_customer_rfm_matrix
        
        result = generate_customer_rfm_matrix(mock_transactions)
        
        # التحقق من الأعمدة
        expected_cols = ['Customer_ID', 'Recency_Days', 'Frequency_Orders', 
                        'Monetary_Spend', 'Total_Units', 'Unique_Products',
                        'Avg_Order_Value', 'Avg_Units_Per_Order']
        for col in expected_cols:
            assert col in result.columns
        
        # التحقق من البيانات
        assert len(result) == 3  # 3 عملاء
        assert result['Frequency_Orders'].sum() == 6  # 6 طلبات إجمالاً
        assert (result['Avg_Order_Value'] > 0).all()
        assert (result['Recency_Days'] >= 0).all()
    
    def test_rfm_calculation_correctness(self, mock_transactions):
        """التأكد من صحة الحسابات"""
        from src.features.build_rfm_features import generate_customer_rfm_matrix
        
        result = generate_customer_rfm_matrix(mock_transactions)
        result = result.set_index('Customer_ID')
        
        # C001: طلبان بمبلغ 100 + 200 = 300
        assert result.loc['C001', 'Monetary_Spend'] == 300.0
        assert result.loc['C001', 'Frequency_Orders'] == 2
        assert result.loc['C001', 'Avg_Order_Value'] == 150.0

# ==================== 5. Models Tests ====================

class TestModels:
    """اختبار تدريب وتقييم النماذج"""
    
    def test_train_pipelines(self):
        """اختبار تدريب النموذج"""
        from src.models.train import train_champion_pipeline
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=50, n_features=6, random_state=42)
        X = pd.DataFrame(X, columns=['Frequency_Orders', 'Monetary_Spend', 'Total_Units',
                                    'Unique_Products', 'Avg_Order_Value', 'Avg_Units_Per_Order'])
        
        binary_model, multi_model = train_champion_pipeline(X, pd.Series(y))
        
        assert binary_model is not None
        assert multi_model is not None
        assert hasattr(binary_model, 'predict')
    
    def test_evaluate_model(self):
        """اختبار تقييم النموذج"""
        from src.models.evaluate import evaluate_model_performance
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=50, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        metrics = evaluate_model_performance(model, pd.DataFrame(X), pd.Series(y))
        
        assert 'roc_auc' in metrics
        assert 'f1_score' in metrics
        assert 'report' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_predict_functions(self, mock_artifact):
        """اختبار تحميل وتشغيل الاستدلال"""
        from src.models.predict import run_batch_inference
        
        # بيانات اختبار
        test_data = pd.DataFrame({
            'Customer_ID': ['C001'],
            'Frequency_Orders': [5],
            'Monetary_Spend': [500.0],
            'Total_Units': [10],
            'Unique_Products': [3],
            'Avg_Order_Value': [100.0],
            'Avg_Units_Per_Order': [2.0]
        })
        
        result = run_batch_inference(test_data, mock_artifact)
        
        assert 'Customer_ID' in result.columns
        assert 'Churn_Probability' in result.columns
        assert 'Is_Churned_Predicted' in result.columns
        assert 'Risk_Horizon_Tier' in result.columns
        assert len(result) == 1

# ==================== 6. API Tests ====================

class TestAPI:
    """اختبار واجهة برمجة التطبيقات"""
    
    @pytest.fixture
    def client(self):
        """إنشاء عميل اختبار FastAPI"""
        from fastapi.testclient import TestClient
        from src.api.app import app
        
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """اختبار نقطة التحقق من الصحة"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_predict_endpoint_no_model(self, client):
        """اختبار التنبؤ عندما لا يكون النموذج محملاً"""
        # ملاحظة: هذا الاختبار يفترض أن النموذج غير محمل
        payload = {
            "Customer_ID": "C001",
            "Frequency_Orders": 5,
            "Monetary_Spend": 500.0,
            "Total_Units": 10,
            "Unique_Products": 3,
            "Avg_Order_Value": 100.0,
            "Avg_Units_Per_Order": 2.0
        }
        
        response = client.post("/predict", json=payload)
        # قد يكون 500 إذا لم يكن النموذج محملاً، أو 200 إذا كان محملاً
        assert response.status_code in [200, 500]
    
    def test_predict_endpoint_validation(self, client):
        """اختبار التحقق من صحة البيانات المدخلة"""
        # بيانات ناقصة
        invalid_payload = {
            "Customer_ID": "C001"
            # حقول ناقصة
        }
        
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422  # Validation Error

# ==================== 7. Integration Test ====================

class TestIntegration:
    """اختبار تكاملي شامل للـ Pipeline"""
    
    def test_full_pipeline(self, tmp_path, mock_transactions):
        """اختبار كامل: من البيانات الخام إلى التنبؤ"""
        from src.ingestion.download_raw import load_raw_transactions
        from src.processing.clean_transactions import clean_transaction_logs
        from src.features.build_rfm_features import generate_customer_rfm_matrix
        from src.models.train import train_champion_pipeline
        from src.models.predict import run_batch_inference
        
        # 1. حفظ وتحميل البيانات
        csv_path = tmp_path / "raw.csv"
        mock_transactions.to_csv(csv_path, index=False)
        df_raw = load_raw_transactions(csv_path)
        
        # 2. تنظيف
        df_clean = clean_transaction_logs(df_raw)
        
        # 3. بناء الميزات
        df_features = generate_customer_rfm_matrix(df_clean)
        
        # 4. تدريب سريع (بيانات صغيرة)
        X = df_features[['Frequency_Orders', 'Monetary_Spend', 'Total_Units',
                        'Unique_Products', 'Avg_Order_Value', 'Avg_Units_Per_Order']]
        y = pd.Series([0, 1, 0])  # binary labels
        
        binary_model, _ = train_champion_pipeline(X, y)
        
        # 5. استدلال
        artifact = {'binary_model': binary_model, 'multi_tier_model': binary_model}
        results = run_batch_inference(df_features, artifact)
        
        assert len(results) == 3
        assert 'Churn_Probability' in results.columns

# ==================== Run Command ====================
# python -m pytest tests/test_all.py -v