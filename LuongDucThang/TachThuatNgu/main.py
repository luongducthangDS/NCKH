import os
import glob
import csv
from test_processor import load_dictionary, replace_with_dict

# Đường dẫn
f_dict = 'F:/Viet_Lao/Vietnamese Conceptizer/Vietnamese Conceptizer/WORDS_WordNet_And_VCL_ALL_sorted.txt'
output_dir = 'F:/Viet_Lao/test/TachThuatNgu/'
replace_dir = 'F:/Viet_Lao/test/TachThuatNgu/ThayThe/'
error_log_dir = 'F:/Viet_Lao/test/TachThuatNgu/ErrorLogs/'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(replace_dir, exist_ok=True)
os.makedirs(error_log_dir, exist_ok=True)

# Load từ điển
dic = load_dictionary(f_dict)
print("Dictionary loaded!")

# Xử lý từng file
os.chdir("F:/Viet_Lao/test/TachThuatNgu/DATA/")
for file in glob.glob("*.txt"):
    f_in = os.path.join("F:/Viet_Lao/test/TachThuatNgu/DATA/", file)
    f_out = os.path.join(output_dir, file + '_WordNet_And_VCL.txt')
    f_log = os.path.join(replace_dir, file + '_WordNet_And_VCL.txt')
    f_error_log = os.path.join(error_log_dir, file + '_ErrorLog.txt')

    print(f"Processing {f_in}")

    # Đọc dữ liệu từ file
    rows = []
    with open(f_in, 'r', encoding="utf-8") as fin:
        lines = fin.readlines()  # Đọc tất cả các dòng vào một danh sách

        # Phân tách mỗi dòng theo dấu tab
        for line in lines:
            row = line.strip().split('\t')  # Loại bỏ ký tự thừa và phân tách theo dấu tab
            rows.append(row)

    # Kiểm tra dữ liệu
    print(f"First 5 rows from {f_in}:")
    for idx, row in enumerate(rows[:5], start=1):
        print(f"Row {idx}: {row}")

    # Kết quả sau xử lý
    result = []
    replated = []
    error_log = []

    # Xử lý từng dòng
    for row_num, row in enumerate(rows, start=1):
        try:
            # Kiểm tra số cột
            if len(row) < 3:
                raise ValueError(f"Insufficient columns: {row}")

            # Xử lý từng cột
            viet1 = replace_with_dict(row[0], dic, replated)
            viet2 = replace_with_dict(row[1], dic, replated)
            score = row[2]
            result.append([viet1, viet2, score])

        except Exception as e:
            error_message = f"Row {row_num}: Error - {row}. Exception: {e}"
            print(error_message)  # In lỗi ra màn hình để kiểm tra
            error_log.append(error_message)


    # Ghi file kết quả
    with open(f_out, 'w', encoding="utf-8", newline='') as fout:
        csv_writer = csv.writer(fout, delimiter='\t')
        csv_writer.writerows(result)

    # Ghi file log các từ đã thay thế
    with open(f_log, 'w', encoding="utf-8") as flog:
        flog.write('\n'.join(replated))

    # Ghi file log lỗi
    if error_log:
        with open(f_error_log, 'w', encoding="utf-8") as ferr:
            for error in error_log:
                ferr.write(f"{error}\n")
        print(f"Error log saved to {f_error_log}")


print("done")
