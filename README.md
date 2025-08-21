# CTAI_MachineLearning



Phân tích chi tiết cuộc thi “Fake or Real: The Impostor Hunt in Texts”
1. Dữ liệu

    Mỗi dòng dữ liệu gồm hai đoạn văn (text₀, text₁) lấy từ tạp chí khoa học The Messenger.

    Cả hai đoạn đều đã được chỉnh sửa bởi mô hình ngôn ngữ lớn (LLM):

        text REAL: được tinh chỉnh để bám sát bài gốc.

        text FAKE: được “nhiễu” để lệch đáng kể khỏi bài gốc.

    Nhiệm vụ: xác định đoạn nào là real trong cặp.

    Bộ file do Kaggle cung cấp:

        train.csv (id, text_0, text_1, label) – 1 = text₁ real, 0 = text₀ real.

    test.csv (id, text_0, text_1) – không có nhãn.

    sample_submission.csv – khung nộp kết quả.

Kích thước (ước tính từ trang mô tả): ~40,000 cặp train; test chia 45% public, 55% private cho leaderboard.
2. Dạng bài toán & chấm điểm

    Thuộc nhóm binary classification trên cặp văn bản: dự đoán nhãn 0/1.

    Input = (text₀, text₁); Output = 0 hoặc 1.

    Metric: Accuracy trên 100% bộ test; Kaggle hiển thị public (45%) và giữ kín private (55%) để giảm overfitting.

Bối cảnh bảo mật: cuộc thi thuộc chuỗi Secure Your AI của ESA; nhấn mạnh rủi ro data poisoning & over-reliance khi dùng LLM trong tác vụ khoa học.
