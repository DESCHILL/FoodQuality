# FoodQuality — Model và website sản phẩm

Đây là gói để nộp và trình bày. Không có Google crawler, giao diện tìm ảnh hoặc phụ thuộc Playwright. Dữ liệu từ tool là đầu vào bên ngoài, không cần cài tool trên máy chạy website.

Nghiên cứu bản Drive, thiết kế hai mô hình, dữ liệu, cross-validation, so sánh thực nghiệm và cách tích hợp web: [báo cáo bảy tiêu chí](docs/model-study.md). Số liệu máy đọc được: [research-results.json](docs/research-results.json).

## 1. Chạy sản phẩm

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe chay_web.py
```

Bản nghiên cứu hai checkpoint: `.venv\Scripts\python.exe chay_web.py --profile dual` (cổng 18502). Mặc định `stable` giữ pipeline hiện tại trên cổng 18501: bản dual tăng nhận diện/độ phủ nhưng test nội bộ có số hỏng→tươi cao hơn, nên chưa tự thay bản đang dùng. Xem bảng so sánh trong báo cáo.

Mở http://127.0.0.1:18501. Gói có trọng số hiện tại trong `models/assets/`; launcher tự cấu hình đường dẫn khi chuyển máy. Dừng website cũ trên cổng 18501 nếu muốn chạy bản đóng gói thay thế. Mã nguồn, trọng số và hash của trọng số được giữ nguyên; không cần huấn luyện lại để trình bày.

## 2. Cách model hoạt động

Ảnh tải lên → chuẩn hóa chiều và màu → nhận diện/khoanh vùng → trích đặc trưng → đánh giá Tươi/Hỏng → hiển thị kết quả.

- `foodquality8/ai_detection/`: YOLO khoanh vùng; hợp nhất nhận diện từ ba nguồn dự đoán bằng điểm đã hiệu chuẩn, chọn thực phẩm có điểm cao nhất. Ba nguồn dự đoán không đồng nghĩa ba lần tìm kiếm Google.
- `foodquality8/ai_quality/`: EfficientNet trích đặc trưng, các vùng ảnh chi tiết 320×320 và đặc trưng ngoại quan; bộ phân loại đánh giá trạng thái theo loại thực phẩm. Khi bằng chứng thiếu hoặc bất đồng, kết quả có thể là Chưa xác định.
- `foodquality8/dataset/`: xử lý dữ liệu nhập, lọc, loại trùng, nhãn và chia tập. Không gọi Google trong quy trình nhập dữ liệu.
- `user_app.py`: giao diện sản phẩm. Thêm dữ liệu không tự thay kết quả của model; phải huấn luyện, đánh giá và kích hoạt phiên bản mới.

Phạm vi: bò, gà, cừu; táo, chuối, cam; cà chua, dưa chuột, cà rốt. Đánh giá ngoại quan không xác nhận an toàn ăn uống. Chi tiết thuật toán và kết quả thực nghiệm nằm trong `docs/`.

## 3. Đưa dữ liệu từ tool vào model

Chép toàn bộ thư mục `output` của tool vào máy chạy model (giữ Excel cạnh thư mục `images`), sau đó:

```powershell
.venv\Scripts\python.exe nhap_du_lieu.py --file "D:\DuLieu\ThucPham.xlsx" --food tomato
```

Lệnh cũng nhận thư mục ảnh có thư mục con `fresh` và `spoiled`. Với Excel nhiều thực phẩm, chạy từng loại bằng `--food`. Trình nhập đọc nhãn và ảnh (ưu tiên ảnh gốc nếu đường dẫn còn tồn tại, dùng ảnh nhúng nếu đã chuyển máy), kiểm tra ảnh, lọc chất lượng kỹ thuật và loại trùng, rồi tạo tập train/val/test theo nhóm. Thiếu lớp hoặc thiếu nhóm độc lập sẽ báo lỗi; không tự bịa thêm ảnh hay nhãn. Ảnh nhúng Excel là ảnh xem trước, vì vậy nên mang theo ảnh gốc khi huấn luyện chính thức.

Đường dẫn dataset được in sau khi nhập. Chất lượng nhãn gợi ý vẫn là thử nghiệm; việc nhập không biến nhãn Google thành nhãn đã kiểm chứng. Dữ liệu mới không sửa checkpoint đang chạy.

## 4. Trình bày huấn luyện hai mô hình

```powershell
.venv\Scripts\python.exe -m foodquality8.train_pair --project . --source "D:\DuLieu\ThucPham.xlsx" --food tomato --run model_runs/demo-tomato --epochs 10 --batch 4
```

Lệnh nhập dữ liệu, tạo vùng gợi ý rồi huấn luyện EfficientNet và YOLO lần lượt. Nhãn/vùng tự động được đánh dấu thử nghiệm. Kết quả nằm trong thư mục run: cấu hình, lịch sử học, chỉ số validation/test và checkpoint. Mỗi lần dùng tên run mới. Đặt file `STOP` trong thư mục run để yêu cầu dừng.

Cặp demo được quản lý trong `models/active.json`; website đang dùng bộ chín loại `models/consensus.json` nên cặp thử nghiệm này không tự thay bộ đang trình bày. Điều này tránh một lần học ít ảnh làm mất khả năng nhận diện chín loại.

Để phát triển bộ chín loại, dùng pipeline `scripts/retrain_robust.py` với dataset đủ lớp và cấu hình `--dataset` chứa `source_base` (thư mục có `split.json`) và `training_base` (thư mục cache/biến thể). Pipeline tạo biến thể chỉ trên tập train, học, hiệu chuẩn, so sánh và chỉ công bố nếu vượt các ngưỡng kiểm tra. Tập test không dùng chọn phiên bản. Không chạy pipeline này với Excel trực tiếp hoặc dataset chỉ một loại.

## 5. Nội dung nộp và tài nguyên

Repository chứa code, trọng số đang dùng, tài liệu và mã kiểm thử cho model + website. Không chứa toàn bộ 16.244 ảnh huấn luyện; nguồn dữ liệu/giấy phép được ghi tại `docs/data-sources.md`. Dataset ảnh cần lưu riêng vì kích thước lớn. Tool thu thập được bàn giao riêng, không nằm trong repository này.
