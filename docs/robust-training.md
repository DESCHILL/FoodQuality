# Huấn luyện với ảnh nhiều điều kiện

Phạm vi gồm táo, chuối, cam, bò, gà, cừu, cà rốt, dưa chuột và cà chua. Hai nhãn nguồn là tươi và hỏng; website vẫn có thể trả về chưa đủ bằng chứng.

## Dữ liệu

Bản `robust9_v1` gồm 6.823 ảnh nguồn (FoodClass V2, AgML và AgriFreshNET) và 4.604 ảnh biến đổi từ tập train. Tổng cộng 11.427 ảnh: train 9.208, validation 1.098, test 1.121. Các ảnh biến đổi giữ nhãn, nguồn, mã nhóm và SHA256 ảnh cha; không được đi vào validation/test. Ảnh biến đổi không làm tăng số mẫu độc lập. Thịt cừu vẫn ít nhóm độc lập.

Đợt mở rộng ngày 18/10/2026 có 16.244 ảnh nguồn: FoodClass V2 13.475, AgML 1.181 và AgriFreshNET 1.588. Thêm 11.159 biến thể train thành 27.403 ảnh, chia train 22.318, validation 2.597, test 2.488. Tập kiểm tra cũ giữ nguyên; ảnh mới được bổ sung vào các tập sau khi lọc trùng. Đây là dữ liệu chuẩn bị huấn luyện, không đồng nghĩa mô hình website đã được thay.

AgriFreshNET bổ sung 957 ảnh đại diện chuối, cam, dưa chuột, loại 2.502 biến thể dư thừa và 45 ảnh gần trùng. Nguồn hiện tải được là Processed Data; các ảnh mới được chọn không có tiền tố tăng cường, nhưng không khẳng định đó là ảnh raw. Ngày chụp làm nhóm bảo thủ và giữ nguyên nhóm cũ. Không đổi nhãn semi-fresh thành fresh hoặc spoiled.

Đợt FoodClass tải 16.300 tệp có kiểm tra CRC, giữ thêm 8.464 ảnh sau khi loại 6.210 ảnh trùng nguồn cũ, 1.584 ảnh dư trong nhóm gần trùng và 42 ảnh xung đột nhãn. Nhóm gần trùng được nối bắc cầu, không chỉ so sánh với ảnh đại diện. Mỗi nhóm mới giữ một ảnh. Gà và cừu không có mẫu bổ sung từ đợt này. Nhóm hash/ngày không chứng minh các vật thể vật lý độc lập; nhãn nguồn chưa được duyệt chuyên gia.

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

Chọn thư mục run mới mỗi lần để không ghi đè checkpoint. Tạo tệp STOP trong thư mục run để ngừng; tiến trình con dừng và không thay mô hình website.

`workspace/dataset-current.json` chọn `source_base`, `training_base`, `reuse_source` và `reuse_training` cho lệnh huấn luyện. Không có cấu hình này thì dùng bộ cũ. Đặc trưng của encoder ImageNet cố định được tái sử dụng theo SHA256 ảnh và đúng công thức tiền xử lý; chỉ ảnh mới chạy lại encoder. Tệp tiến độ cho phép tiếp tục lần trích đặc trưng bị ngừng. Không chạy hai tác vụ ghi đặc trưng vào cùng một thư mục đồng thời.

Trích đặc trưng trên CPU dùng bố trí bộ nhớ channels-last, giữ nguyên kích thước, trọng số và phép tiền xử lý. Khi đánh giá, ảnh sạch có thể tái sử dụng đặc trưng nếu đúng manifest, công thức và encoder ImageNet cố định; YOLO vẫn chạy trên ảnh. Ảnh xoay/mờ/che vẫn tính đặc trưng mới. Quy trình tự đối chiếu kết quả trực tiếp và kết quả dùng cache trên một ảnh validation của mỗi loại/trạng thái, yêu cầu cùng nhãn và điểm tương đương trong sai số số học trước khi kiểm tra toàn bộ tập.

Dữ liệu được nhập từ bên ngoài theo hướng dẫn trong README. Repository này không chứa tool thu thập. Với dataset chín loại đã có split.json, dùng scripts/build_robust_dataset.py để tạo biến thể và scripts/retrain_robust.py để huấn luyện, đánh giá.

## Kiểm tra và phát hành

Ứng viên phải tăng macro F1 trên validation, không giảm nhận dạng loại và recall lớp hỏng; mức giảm F1 trên nguồn cũ tối đa 0,5 điểm phần trăm. Sau đó chạy toàn bộ validation qua ba bộ nhận diện và một tập phân tầng với ảnh xoay 22°, tối/mờ, che một góc. Ảnh che khuất chỉ chấm nhận dạng loại vì dấu hiệu hỏng có thể bị che.

Khi mở rộng dữ liệu, kiểm tra thêm đúng tập validation 1.098 ảnh đã khóa: phải giữ đủ SHA256 ảnh, không giảm độ chính xác loại và không giảm macro F1 quá 0,5 điểm phần trăm. Báo cáo tách tập mới bổ sung khỏi tập cũ. Test từng được dùng trong phát triển nên vẫn là đánh giá nội bộ, chưa thay thế bộ ảnh thực tế độc lập.

Chỉ cập nhật website khi cả hai kiểm tra đạt; không chỉ tăng tỷ lệ trả về “chưa rõ” để làm đẹp độ chính xác. Kiểm tra checksum checkpoint/cấu hình, lưu bản khôi phục rồi thay registry nguyên tử. Các số đo nội bộ không chứng minh độ chính xác ngoài thực tế hoặc an toàn ăn uống.

Tham khảo phép biến đổi từ [Torchvision](https://docs.pytorch.org/vision/stable/transforms.html) và hướng đánh giá độ bền của [Google Research AugMix](https://github.com/google-research/augmix). Quy trình này dùng biến đổi nhẹ có truy vết; không tuyên bố đã triển khai hoặc đạt kết quả của AugMix.
