"""Import an external spreadsheet or labelled image directory without crawling."""
import argparse
from pathlib import Path
from foodquality8.dataset.auto_workflow import import_local
from foodquality8.dataset.quick_dataset import prepare_google

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Nhập dữ liệu bên ngoài thành dataset phân loại thử nghiệm.')
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--food', required=True, choices=['beef','chicken','mutton','apple','banana','orange','tomato','cucumber','carrot'])
    args = parser.parse_args()
    folder = import_local(ROOT, args.file, args.food)
    release = prepare_google(ROOT, args.food, source_folder=folder, required_stages=['fresh','spoiled'])
    print(f'Dataset đã tạo: {release}')
    print('Chưa huấn luyện hoặc thay mô hình website. Nhãn nhập vào cần được kiểm tra trước khi đánh giá chất lượng.')


if __name__ == '__main__':
    main()
