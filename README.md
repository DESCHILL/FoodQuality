# FoodQuality

Ứng dụng thu thập ảnh thực phẩm, huấn luyện mô hình nhận diện và đánh giá ngoại quan Tươi/Hỏng.

## Chạy ứng dụng

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[ai]"
.venv\Scripts\python.exe launch_user_app.py
```

Website: `http://127.0.0.1:18501`. Công cụ quản lý dữ liệu và huấn luyện: `Mo-GUI-FoodQuality.bat`.

Weights, ảnh và cấu hình chứa đường dẫn cục bộ không đưa vào Git. Cần registry phù hợp trong `models/` trước khi suy luận; giao diện báo rõ khi thiếu model.

## Cấu trúc

Mã nghiệp vụ được chia thành đúng **3 phần**:

| Phần | Thư mục | Chức năng |
| --- | --- | --- |
| Dataset dùng chung | `foodquality8/dataset/` | Thu thập Google, nhập dữ liệu, lọc ảnh, nhãn, Excel, chia tập và gộp chỉ mục |
| AI nhận diện | `foodquality8/ai_detection/` | Huấn luyện YOLO, khoanh vùng, hợp nhất kết quả nhận diện và chạy suy luận |
| AI chất lượng | `foodquality8/ai_quality/` | EfficientNet, đặc trưng nhiều vùng ảnh, phân loại Tươi/Hỏng, đánh giá và chọn model |

Các file Python cũ trực tiếp trong `foodquality8/` và `image_search.py` chỉ chuyển tiếp import để các lệnh chạy, GUI và mã tích hợp cũ tiếp tục hoạt động. `scripts/` chứa lệnh chạy; `tests/` chứa kiểm thử; `user_app.py` và các GUI là giao diện dùng ba phần trên. Không có bản sao thuật toán riêng trong các file chuyển tiếp.

### Dataset FoodQuality

Các bộ ảnh được sử dụng như một dataset logic tên **FoodQuality**. Chỉ mục chung giữ đường dẫn đến ảnh gốc, nhãn, nhóm ảnh trùng và tập train/val/test. Metadata giữ xuất xứ thực tế; ảnh nhập từ dataset khác không được ghi thành ảnh do Google collector tải về.

```powershell
.venv\Scripts\python.exe -m foodquality8.dataset.unified --manifest workspace/source9_v5/split.json --output workspace/dataset-unified
```

Có thể thêm nhiều `--manifest` để gộp các chỉ mục cùng cấu trúc. Lệnh kiểm tra SHA-256 từng ảnh, loại bản ghi trùng hoàn toàn và từ chối xung đột nhãn/nguồn hoặc nhóm ảnh nằm ở nhiều tập. Thư mục đích phải chưa tồn tại để tránh ghi đè dữ liệu. `split.json` dùng cùng định dạng với pipeline huấn luyện hiện tại; `dataset.json` lưu thống kê và nguồn gốc. Chỉ mục tham chiếu ảnh gốc, vì vậy khi chuyển máy cần mang theo ảnh và cập nhật đường dẫn. Các model, cache và cấu hình huấn luyện đang hoạt động được giữ nguyên.

Phạm vi Tươi/Hỏng: bò, gà, cừu; táo, chuối, cam; cà chua, dưa chuột, cà rốt. Nhận dạng và trạng thái là hai bài toán riêng; ảnh không đủ bằng chứng có thể trả Chưa xác định. Kết quả ngoại quan không xác nhận an toàn ăn uống.

## Kiểm thử

```powershell
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -q
```

Các nhánh chức năng được kiểm thử trước khi merge vào `main`. Báo cáo thực nghiệm ghi rõ tập đánh giá, độ phủ và giới hạn.

Xem [quy trình phát triển](docs/development.md), [thuật toán](docs/algorithms.md) và [kết quả đánh giá](docs/evaluation.md).

Quy trình tự tạo biến thể ảnh, học hai vùng 320×320 và kiểm tra trước khi cập nhật website: [huấn luyện nhiều điều kiện](docs/robust-training.md).

## Nguồn

- [FoodClass V2](https://data.mendeley.com/datasets/5fjkkv5xgt/2), CC BY 4.0.
- [Fresh and Rotten Fruits](https://data.mendeley.com/datasets/bdd69gyhv8/1), CC BY 4.0, bản raw Project-AgML.
- [AgriFreshNET V1](https://data.mendeley.com/datasets/42m5tb7yv9/1), CC BY 4.0; ảnh cà chua tươi/hỏng, giữ riêng các ngày chụp giữa các tập.
- [SIC-Food-Handover](https://github.com/manhhuu13112005/SIC-Food-Handover), checkpoint dùng đối chiếu cục bộ. Giấy phép từng dataset/weight được quản lý riêng; repository không cấp lại quyền cho tài nguyên bên thứ ba.
