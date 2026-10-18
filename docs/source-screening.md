# Sàng lọc nguồn bổ sung ngày 18/10/2026

## Nguồn đã nhập

FoodClass V2, AgML và AgriFreshNET giữ nhãn của tác giả. Tệp gốc có checksum, ảnh biến thể có ảnh cha, các nhóm dữ liệu cũ giữ nguyên tập. Báo cáo số lượng và các loại trừ nằm trong `data-report.json` của từng phiên bản dataset. Không coi ảnh sau tiền xử lý hoặc ảnh khác góc của cùng mẫu là bằng chứng về số vật thể độc lập.

## Nguồn đã xem nhưng chưa nhập

- [Good and Bad Classification of Mutton](https://data.mendeley.com/datasets/9b5gzz2bp9/1): mô tả ghi **Capra hircus** (dê), dù tiêu đề dùng mutton. Không tự ánh xạ bộ này thành thịt cừu. Điều này chưa chứng minh lớp Mutton trong FoodClass là dê; danh tính loài của lớp hiện có vẫn chưa được xác minh độc lập.
- [Hourly Meat Freshness](https://data.mendeley.com/datasets/4tj9t3n6vj/1): nhãn là loài và giờ bảo quản. Không tự chuyển các mốc giờ thành nhãn tươi/hỏng vì trang nguồn chưa cung cấp ánh xạ nhị phân có căn cứ.
- [Leak-Free Meat Freshness Benchmark](https://github.com/AhmetCannnn/leak-free-meat-freshness-benchmark): bản làm sạch có 1.849 ảnh, loại tăng cường ngoại tuyến và gộp các lớp bò 36/48 giờ trùng byte. Đây là nguồn tham khảo kiểm toán có ích, nhưng vẫn là nhãn thời gian, không phải nhãn nhị phân theo yêu cầu hiện tại. Kết quả của repo này không phải kết quả của FoodQuality.

Muốn bổ sung nguồn thịt mới cần xác minh cả loài, cách xác định trạng thái và nhóm mẫu chụp. Chỉ trùng tên nhãn hoặc có số lượng ảnh lớn chưa đủ để nhập vào bộ train đang dùng.
