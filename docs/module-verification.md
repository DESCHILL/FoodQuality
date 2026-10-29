# Kiểm tra cấu trúc ba module

Ngày kiểm tra: 29/10/2026.

- Dataset: `foodquality8/dataset`.
- AI nhận diện: `foodquality8/ai_detection`.
- AI chất lượng: `foodquality8/ai_quality`.

Các đường import cũ chuyển tiếp đến cùng đối tượng module, giữ tương thích với GUI, CLI, checkpoint và các điểm tích hợp hiện có. Cache của website lấy fingerprint từ file triển khai thực tế trong module mới.

## Kết quả tại máy phát triển

- 83 kiểm thử đơn vị/tích hợp thành công.
- 18 ảnh validation, mỗi cặp thực phẩm/trạng thái một ảnh: toàn bộ kết quả suy luận trước và sau tách module bằng nhau chính xác, bao gồm các điểm số. Đây là kiểm tra hồi quy, không phải phép đo độ chính xác mới.
- Kiểm tra website tự phân tích ảnh tải lên và sử dụng kết quả ba nguồn: thành công. Ảnh cà chua hồi quy được nhận diện là cà chua và không bị trả về hỏng; chạy lại trang giữ nguyên kết quả cache.
- CLI huấn luyện đơn, huấn luyện cặp và quản lý dataset giữ nguyên các tham số.
- Chỉ mục FoodQuality chung: 16.244 ảnh; train 11.159, validation 2.597, test 2.488. SHA-256 của từng ảnh được xác minh khi tạo chỉ mục; nhóm ảnh không vượt ranh giới các tập.
- Xuất xứ: FoodClass V2 13.475 ảnh, AgML raw 1.181 ảnh, AgriFreshNET V1 1.588 ảnh. Không thay đổi nhãn hoặc ghi lại xuất xứ thành Google.

Không huấn luyện lại hoặc thay checkpoint trong thay đổi cấu trúc này. Các phép kiểm tra trên xác nhận tính tương thích; không mở rộng tuyên bố về khả năng đánh giá độ an toàn thực phẩm.
