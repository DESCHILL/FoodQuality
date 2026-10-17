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

- `image_search.py`: tìm ảnh Google, tải, lọc và xuất Excel.
- `foodquality8/training.py`: huấn luyện EfficientNet và YOLO.
- `foodquality8/train_pair.py`: quy trình huấn luyện hai mô hình.
- `foodquality8/`: quản lý dữ liệu, suy luận và kiểm tra model.
- `scripts/`: nhập dataset, huấn luyện, so sánh và đánh giá.
- `tests/`: kiểm thử dữ liệu, huấn luyện và giao diện.
- `user_app.py`: website tra cứu.

Phạm vi Tươi/Hỏng: bò, gà, cừu; táo, chuối, cam; cà chua, dưa chuột, cà rốt. Nhận dạng và trạng thái là hai bài toán riêng; ảnh không đủ bằng chứng có thể trả Chưa xác định. Kết quả ngoại quan không xác nhận an toàn ăn uống.

## Kiểm thử

```powershell
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -q
```

Các nhánh chức năng được kiểm thử trước khi merge vào `main`. Báo cáo thực nghiệm ghi rõ tập đánh giá, độ phủ và giới hạn.

Xem [quy trình phát triển](docs/development.md), [thuật toán](docs/algorithms.md) và [kết quả đánh giá](docs/evaluation.md).

## Nguồn

- [FoodClass V2](https://data.mendeley.com/datasets/5fjkkv5xgt/2), CC BY 4.0.
- [Fresh and Rotten Fruits](https://data.mendeley.com/datasets/bdd69gyhv8/1), CC BY 4.0, bản raw Project-AgML.
- [SIC-Food-Handover](https://github.com/manhhuu13112005/SIC-Food-Handover), checkpoint dùng đối chiếu cục bộ. Giấy phép từng dataset/weight được quản lý riêng; repository không cấp lại quyền cho tài nguyên bên thứ ba.
