from pyspark.sql import SparkSession
from config import DB_CONFIG

print("--- BƯỚC 3: ĐẨY DỮ LIỆU SẠCH VÀ BÁO CÁO VÀO MYSQL ---")
spark = SparkSession.builder \
    .appName("Cardio_Export_MySQL") \
    .config("spark.jars.packages", "mysql:mysql-connector-java:8.0.33") \
    .getOrCreate()

# Danh sách các bảng Parquet đang nằm trong HDFS
tables_to_export = {
    "cardio_cleaned": "hdfs://localhost:9000/cardio_cleaned.parquet",
    "risk_age": "hdfs://localhost:9000/risk_age.parquet",
    "risk_bmi": "hdfs://localhost:9000/risk_bmi.parquet",
    "risk_bp": "hdfs://localhost:9000/risk_bp.parquet",
    "risk_gender": "hdfs://localhost:9000/risk_gender.parquet",
    "risk_cholesterol": "hdfs://localhost:9000/risk_cholesterol.parquet",
    "risk_gluc": "hdfs://localhost:9000/risk_gluc.parquet",
    "risk_smoke": "hdfs://localhost:9000/risk_smoke.parquet",
    "risk_alco": "hdfs://localhost:9000/risk_alco.parquet",
    "risk_active": "hdfs://localhost:9000/risk_active.parquet"
}

print(f"Bắt đầu kết nối đến MySQL tại: {DB_CONFIG['url']}")

for table_name, hdfs_path in tables_to_export.items():
    print(f">> Đang đẩy bảng '{table_name}' vào MySQL...")
    try:
        df = spark.read.parquet(hdfs_path)
        df.write.jdbc(
            url=DB_CONFIG["url"], 
            table=table_name, 
            mode="overwrite", 
            properties=DB_CONFIG
        )
        print(f"   [OK] Đã lưu thành công: {table_name}")
    except Exception as e:
        print(f"   [LỖI] Không thể lưu {table_name}. Lỗi: {e}")

print("\n🎉 HOÀN TẤT ĐẨY DỮ LIỆU!")