from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round

print("--- BƯỚC 1: LÀM SẠCH VÀ CHUẨN BỊ DỮ LIỆU ---")
# Khởi tạo PySpark 
spark = SparkSession.builder.appName("Cardio_Clean_Data").getOrCreate()

print("1. Đang đọc dữ liệu gốc từ HDFS...")
df = spark.read.csv("hdfs://localhost:9000/cardio_train.csv", header=True, inferSchema=True, sep=";")

print("2. Đang xử lý nhiễu, tính tuổi (năm) và BMI...")
df_cleaned = df.withColumn("age_years", round(col("age") / 365.25).cast("integer")) \
    .withColumn("bmi", round(col("weight") / ((col("height") / 100) ** 2), 2)) \
    .filter(
        (col("ap_hi") >= 70) & (col("ap_hi") <= 240) &
        (col("ap_lo") >= 50) & (col("ap_lo") <= 150) &
        (col("ap_hi") > col("ap_lo")) &
        (col("height") >= 100) & (col("height") <= 220) &
        (col("weight") >= 30) & (col("weight") <= 250)
    ).drop("id", "age")

print(f"-> Dữ liệu gốc: {df.count()} hồ sơ")
print(f"-> Dữ liệu sau làm sạch: {df_cleaned.count()} hồ sơ")

print("\n-> MẪU DỮ LIỆU SẠCH:")
df_cleaned.show(5)

print("3. Đang lưu dữ liệu sạch xuống HDFS (Định dạng Parquet)...")
# Lưu đè dữ liệu sạch vào HDFS
df_cleaned.write.mode("overwrite").parquet("hdfs://localhost:9000/cardio_cleaned.parquet")

print("🎉 HOÀN TẤT! File Parquet đã sẵn sàng cho bước phân tích.")