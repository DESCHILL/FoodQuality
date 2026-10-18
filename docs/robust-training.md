# Huấn luyện với ảnh nhiều điều kiện

Phạm vi gồm táo, chuối, cam, bò, gà, cừu, cà rốt, dưa chuột và cà chua. Hai nhãn nguồn là tươi và hỏng; website vẫn có thể trả về chưa đủ bằng chứng.

## Dữ liệu

Bản `robust9_v1` gồm 6.823 ảnh nguồn (FoodClass V2, AgML và AgriFreshNET) và 4.604 ảnh biến đổi từ tập train. Tổng cộng 11.427 ảnh: train 9.208, validation 1.098, test 1.121. Các ảnh biến đổi giữ nhãn, nguồn, mã nhóm và SHA256 ảnh cha; không được đi vào validation/test. Ảnh biến đổi không làm tăng số mẫu độc lập. Thịt cừu vẫn ít nhóm độc lập.

Phép biến đổi gồm lật ngang, xoay nhẹ có mở rộng khung để giữ nguyên vật thể, điều chỉnh sáng nhỏ và mờ nhẹ. Không vẽ thêm mốc hoặc đổi màu để tạo nhãn hỏng. Phép xoay 2D không thay thế ảnh thật chụp mặt bên hay mặt sau.

Google tự luân phiên truy vấn tươi/hỏng bằng tiếng Việt, tiếng Anh và các góc chụp/bối cảnh. Đây là nguồn ảnh ứng viên; từ khóa tìm kiếm không chứng minh trạng thái, quyền sử dụng hoặc độ chính xác nhãn. Không tự nhập ảnh Google chưa xác minh vào bộ nguồn có nhãn.

## Hai vùng ảnh

Bộ FoodQuality mới dùng hai vùng 320×320: toàn ảnh giữ tỷ lệ và vùng giữa giữ 88% chiều rộng/cao. Mỗi vùng có 102.400 pixel thay cho 50.176 pixel của 224×224 (2,04 lần). Hai vùng lấy trung bình đặc trưng; thống kê màu lấy trên toàn ảnh gốc để không coi phần đệm là thực phẩm.

SIC EfficientNet vẫn dùng 224×224 và SIC YOLO dùng 640, đúng cấu hình của các checkpoint có sẵn. Không đổi kích thước các mô hình này rồi tiếp tục dùng điểm hiệu chỉnh cũ. Tăng độ phân giải đầu vào không phục hồi chi tiết đã mất trong ảnh nhỏ hoặc mờ.

Trạng thái thực phẩm có thể kết hợp hai đầu phân loại cũ/mới, với một trọng số chung chọn trên validation để giữ recall lớp hỏng. Đầu cũ sử dụng đặc trưng 224 đã có từ bộ SIC; không chạy lại encoder trong luồng ba bộ nhận diện. Nhận dạng loại vẫn dùng hai vùng 320; đây không phải thêm một bộ nhận diện thứ tư. Mọi đầu phân loại và trọng số kết hợp được lưu trong checkpoint.

## Chạy

Trong giao diện, chọn **Dataset kết hợp SIC** và bấm **HUẤN LUYỆN AI**. Quy trình tự tạo dữ liệu, trích đặc trưng, học, so sánh, hiệu chỉnh điểm và kiểm tra luồng ba bộ nhận diện. Giữ detector SIC đã huấn luyện. Không thêm nút thao tác.

Chạy trực tiếp:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/retrain_robust.py --run model_runs/food9/new-run
```

Chọn thư mục run mới mỗi lần để không ghi đè checkpoint. Nút Ngừng tạo tệp STOP; tiến trình con dừng và không thay mô hình website.

## Kiểm tra và phát hành

Ứng viên phải tăng macro F1 trên validation, không giảm nhận dạng loại và recall lớp hỏng; mức giảm F1 trên nguồn cũ tối đa 0,5 điểm phần trăm. Sau đó chạy toàn bộ validation qua ba bộ nhận diện và một tập phân tầng với ảnh xoay 22°, tối/mờ, che một góc. Ảnh che khuất chỉ chấm nhận dạng loại vì dấu hiệu hỏng có thể bị che.

Chỉ cập nhật website khi cả hai kiểm tra đạt; không chỉ tăng tỷ lệ trả về “chưa rõ” để làm đẹp độ chính xác. Kiểm tra checksum checkpoint/cấu hình, lưu bản khôi phục rồi thay registry nguyên tử. Các số đo nội bộ không chứng minh độ chính xác ngoài thực tế hoặc an toàn ăn uống.

Tham khảo phép biến đổi từ [Torchvision](https://docs.pytorch.org/vision/stable/transforms.html) và hướng đánh giá độ bền của [Google Research AugMix](https://github.com/google-research/augmix). Quy trình này dùng biến đổi nhẹ có truy vết; không tuyên bố đã triển khai hoặc đạt kết quả của AugMix.
