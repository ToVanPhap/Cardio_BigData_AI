from pyspark.sql import SparkSession

print("--- BƯỚC 2: PHÂN TÍCH NGHIỆP VỤ TOÀN DIỆN BẰNG SPARK SQL ---")
spark = SparkSession.builder.appName("Cardio_Analyze_Data").getOrCreate()

print("1. Đang tải dữ liệu sạch từ HDFS...")
df_cleaned = spark.read.parquet("hdfs://localhost:9000/cardio_cleaned.parquet")
df_cleaned.createOrReplaceTempView("cardio_table")

print("\n--- PHÂN TÍCH 1: RỦI RO THEO NHÓM TUỔI ---")
risk_by_age = spark.sql("""
    SELECT 
        CASE 
            WHEN age_years < 40 THEN '1. Duoi 40'
            WHEN age_years BETWEEN 40 AND 50 THEN '2. 40-50 tuoi'
            WHEN age_years BETWEEN 51 AND 60 THEN '3. 51-60 tuoi'
            ELSE '4. Tren 60'
        END AS age_group,
        COUNT(*) as total_people,
        SUM(cardio) as sick_people,
        ROUND((SUM(cardio) / COUNT(*)) * 100, 2) as risk_percentage
    FROM cardio_table
    GROUP BY 
        CASE 
            WHEN age_years < 40 THEN '1. Duoi 40'
            WHEN age_years BETWEEN 40 AND 50 THEN '2. 40-50 tuoi'
            WHEN age_years BETWEEN 51 AND 60 THEN '3. 51-60 tuoi'
            ELSE '4. Tren 60'
        END
    ORDER BY age_group ASC
""")
risk_by_age.show()

print("\n--- PHÂN TÍCH 2: RỦI RO THEO CHỈ SỐ THỂ TRỌNG (BMI) ---")
risk_by_bmi = spark.sql("""
    SELECT 
        CASE 
            WHEN bmi < 18.5 THEN '1. Thieu can'
            WHEN bmi >= 18.5 AND bmi < 25 THEN '2. Binh thuong'
            WHEN bmi >= 25 AND bmi < 30 THEN '3. Thua can'
            ELSE '4. Beo phi'
        END AS bmi_category,
        COUNT(*) as total_people,
        SUM(cardio) as sick_people,
        ROUND((SUM(cardio) / COUNT(*)) * 100, 2) as risk_percentage
    FROM cardio_table
    GROUP BY 
        CASE 
            WHEN bmi < 18.5 THEN '1. Thieu can'
            WHEN bmi >= 18.5 AND bmi < 25 THEN '2. Binh thuong'
            WHEN bmi >= 25 AND bmi < 30 THEN '3. Thua can'
            ELSE '4. Beo phi'
        END
    ORDER BY bmi_category ASC
""")
risk_by_bmi.show()

print("\n--- PHÂN TÍCH 3: RỦI RO THEO TÌNH TRẠNG HUYẾT ÁP ---")
risk_by_bp = spark.sql("""
    SELECT 
        CASE 
            WHEN ap_hi < 120 AND ap_lo < 80 THEN '1. HA Binh thuong'
            WHEN (ap_hi BETWEEN 120 AND 129) AND ap_lo < 80 THEN '2. HA Hoi cao'
            WHEN (ap_hi BETWEEN 130 AND 139) OR (ap_lo BETWEEN 80 AND 89) THEN '3. Tang HA Do 1'
            ELSE '4. Tang HA Do 2'
        END AS bp_category,
        COUNT(*) as total_people,
        SUM(cardio) as sick_people,
        ROUND((SUM(cardio) / COUNT(*)) * 100, 2) as risk_percentage
    FROM cardio_table
    GROUP BY 
        CASE 
            WHEN ap_hi < 120 AND ap_lo < 80 THEN '1. HA Binh thuong'
            WHEN (ap_hi BETWEEN 120 AND 129) AND ap_lo < 80 THEN '2. HA Hoi cao'
            WHEN (ap_hi BETWEEN 130 AND 139) OR (ap_lo BETWEEN 80 AND 89) THEN '3. Tang HA Do 1'
            ELSE '4. Tang HA Do 2'
        END
    ORDER BY bp_category ASC
""")
risk_by_bp.show()

print("\n--- PHÂN TÍCH 4: KHÁM PHÁ CÁC YẾU TỐ SINH HOẠT VÀ LÂM SÀNG KHÁC ---")
factors = {
    "gender": "Gioi tinh (1: Nu, 2: Nam)",
    "cholesterol": "Cholesterol (1: B.Thuong, 2: Cao, 3: Rat cao)",
    "gluc": "Duong huyet (1: B.Thuong, 2: Cao, 3: Rat cao)",
    "smoke": "Hut thuoc (0: Khong, 1: Co)",
    "alco": "Uong ruou (0: Khong, 1: Co)",
    "active": "Van dong (0: Khong, 1: Co)"
}

# Tạo dictionary chứa các DataFrame để lát lưu hàng loạt
factor_dfs = {}

for col_name, description in factors.items():
    print(f"\n>> Yếu tố: {description}")
    query = f"""
        SELECT 
            {col_name} AS category,
            COUNT(*) as total_people,
            SUM(cardio) as sick_people,
            ROUND((SUM(cardio) / COUNT(*)) * 100, 2) as risk_percentage
        FROM cardio_table
        GROUP BY {col_name}
        ORDER BY category ASC
    """
    result_df = spark.sql(query)
    result_df.show()
    factor_dfs[col_name] = result_df

print("\n5. Đang lưu TOÀN BỘ báo cáo xuống HDFS (Định dạng Parquet)...")
risk_by_age.write.mode("overwrite").parquet("hdfs://localhost:9000/risk_age.parquet")
risk_by_bmi.write.mode("overwrite").parquet("hdfs://localhost:9000/risk_bmi.parquet")
risk_by_bp.write.mode("overwrite").parquet("hdfs://localhost:9000/risk_bp.parquet")

for col_name, df_result in factor_dfs.items():
    df_result.write.mode("overwrite").parquet(f"hdfs://localhost:9000/risk_{col_name}.parquet")

print("🎉 HOÀN TẤT! Đã rà soát và lưu lại toàn bộ các yếu tố vào HDFS.")