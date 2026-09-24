from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

print("--- BƯỚC 4: HUẤN LUYỆN AI DỰ ĐOÁN BỆNH TIM (RANDOM FOREST) ---")

# 1. Khởi tạo PySpark
spark = SparkSession.builder.appName("Cardio_AI_Training").getOrCreate()
spark.sparkContext.setLogLevel("ERROR") # Tắt bớt log rác để màn hình sạch đẹp

# 2. Tải dữ liệu siêu tốc từ HDFS
print("1. Đang tải dữ liệu sạch từ HDFS...")
df = spark.read.parquet("hdfs://localhost:9000/cardio_cleaned.parquet")

# Lấy danh sách các cột làm "đầu vào" (Trừ cột cardio là "đầu ra" cần dự đoán)
feature_columns = [col_name for col_name in df.columns if col_name != 'cardio']

# Gộp tất cả các cột thành một cột vector duy nhất mang tên 'features' (Chuẩn của PySpark ML)
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data_ml = assembler.transform(df)

# 3. Chia dữ liệu: 80% để học (Train), 20% để thi (Test)
print("2. Đang chia tập dữ liệu (80% Train, 20% Test)...")
train_data, test_data = data_ml.randomSplit([0.8, 0.2], seed=42)
print(f"   -> Tập huấn luyện: {train_data.count()} hồ sơ")
print(f"   -> Tập kiểm thử: {test_data.count()} hồ sơ")

# 4. Huấn luyện mô hình Random Forest (Rừng ngẫu nhiên với 100 cây quyết định)
print("\n3. Đang huấn luyện Rừng ngẫu nhiên (Random Forest)... (Vui lòng đợi vài giây)")
rf = RandomForestClassifier(featuresCol="features", labelCol="cardio", numTrees=100, maxDepth=10, seed=42)
model = rf.fit(train_data)

# 5. Làm bài thi và Chấm điểm (Đánh giá mô hình)
print("4. Đang làm bài test và chấm điểm AI...\n")
predictions = model.transform(test_data)

# Đánh giá Độ chính xác tổng thể (Accuracy)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="cardio", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_acc.evaluate(predictions)

# Đánh giá Diện tích dưới đường cong ROC (AUC - Chỉ số rất quan trọng trong y khoa)
evaluator_auc = BinaryClassificationEvaluator(labelCol="cardio", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator_auc.evaluate(predictions)

print("="*50)
print(" BÁO CÁO KẾT QUẢ HUẤN LUYỆN TRÍ TUỆ NHÂN TẠO ")
print("="*50)
print(f"👉 Độ chính xác tổng thể (Accuracy) : {accuracy * 100:.2f}%")
print(f"👉 Chỉ số tin cậy y khoa (AUC-ROC)  : {auc * 100:.2f}%")
print("="*50)

# 6. Trích xuất Tầm quan trọng của các đặc trưng (Feature Importance)
print("\nAI ĐÃ HỌC ĐƯỢC RẰNG CÁC YẾU TỐ SAU ĐÂY GÂY BỆNH NHIỀU NHẤT:")
importances = model.featureImportances.toArray()
feature_importance_list = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)

for i, (feature, importance) in enumerate(feature_importance_list):
    print(f"{i+1}. {feature.ljust(15)} : {importance * 100:.2f}%")

print("\n🎉 HOÀN TẤT! Bạn đã đào tạo thành công một AI chẩn đoán y khoa.")

print("\n7. Đang lưu 'não' AI xuống kho HDFS...")
model.write().overwrite().save("hdfs://localhost:9000/random_forest_model")
print("Đã lưu AI thành công! Sẵn sàng cho triển khai thực tế.")