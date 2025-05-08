import fun
import os

output_dir_raw = "RawData/Kelas 1 40/"
output_dir = "Data/Kelas 1 40"
slice_cols = 5
slice_rows = 3
margin_x = 50
margin_y = 78

path = "F:/TUGAS/UNHAS/TESIS/Data/Kelas 1 40"

# buat kotak utama
# file_names = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]

# for file in file_names:
#     fun.crop_largest_box(f"{path}/{file}", file, output_dir_raw)


# potong kotak
# file_names = [file for file in os.listdir(output_dir_raw) if os.path.isfile(os.path.join(output_dir_raw, file))]

# for index, file in enumerate(file_names):
#     if index == 5:
#         break
#     fun.slice_image_with_margin(f"{output_dir_raw}/{file}", index, output_dir, slice_cols, slice_rows, margin_x, margin_y, (35,35))


file_names = [file for file in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, file))]
print(file_names, len(file_names))

file1 = file_names[0]

x = fun.load_pixel(f"{output_dir}/{file1}")

print(x.shape)




