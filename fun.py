from PIL import Image
import os
import glob
import cv2
import numpy as np


def slice_image_with_margin(
    image_path,
    index,
    output_dir="saved",
    slice_cols=5,
    slice_rows=3,
    margin_x=50,
    margin_y=78,
    target_size = (30,30)
):
    img = Image.open(image_path)
    width, height = img.size

    slice_width = width // slice_cols
    slice_height = height // slice_rows

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # for file in glob.glob(f"{output_dir}/{name}/*.png"):
    #     os.remove(file)

    file_names = [str(i) for i in range(1, 10)] + [
        "0",
        "plus",
        "minus",
        "multiplication",
        "division",
        "equal",
    ]

    counter = 1
    for row in range(slice_rows):
        for col in range(slice_cols):
            left = col * slice_width + margin_x
            upper = row * slice_height + margin_y + 10
            right = (col + 1) * slice_width - margin_x
            lower = (row + 1) * slice_height - margin_y

            left = max(0, left)
            upper = max(0, upper)
            right = min(width, right)
            lower = min(height, lower)

            sliced_img = img.crop((left, upper, right, lower))
            
            sliced_img = sliced_img.resize(target_size)
            sliced_img = sliced_img.convert("L")
            
            sliced_img.save(f"{output_dir}/{file_names[counter-1]}_{index + 1}.png")
            counter += 1

    print("Gambar berhasil dipotong dengan margin berbeda!")

def crop_largest_box(image_path, name, output_dir="saved"):
    # Membaca gambar
    img = cv2.imread(image_path)
    if img is None:
        print("Gambar tidak ditemukan")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Konversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding biner
    _, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
    # _, thresh = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY_INV)

    # Mencari kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_contour = None

    # Mencari kontur persegi terbesar
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.0609 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Hanya kontur dengan 4 sudut
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                best_contour = approx

    if best_contour is not None:
        # Dapatkan koordinat bounding box
        x, y, w, h = cv2.boundingRect(best_contour)

        # Potong gambar
        cropped = img[y : y + h, x : x + w]

        # Simpan hasil
        cv2.imwrite(f"{output_dir}/{name}", cropped)
        # print("Berhasil disimpan sebagai 'hasil_potongan.jpg'")
    else:
        print("Tidak ditemukan kotak")


def load_pixel(image_path, threshold=None, convert=True):
    img = Image.open(image_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to numpy array (uint8 by default)
    pixel_array = np.array(img, dtype=np.uint8) if convert else np.array(img)
    
    # Determine threshold (use parameter if provided, else auto-calculate)
    teta = threshold if threshold is not None else (np.max(pixel_array) + np.min(pixel_array)) // 2
    
    # Binarize: values >= teta become 0 (background), others 1 (foreground)
    binary_array = np.where(pixel_array >= teta, 0.0, 1.0).astype(np.float32)
    
    return binary_array.flatten()

def load_pixel2(image_path, threshold=247, convert = True):
    img = Image.open(image_path)
    
    # pixel_array = np.array(img)
        
    # if convert:
    pixel_array = np.array(img, dtype=np.uint8)
    p = pixel_array.flatten()
    teta = (max(p) + min(p))/2
    
    binary_array = np.where(pixel_array >= teta, 0.0, 1.0).astype(np.float32)
    return binary_array.flatten()
    
    # return pixel_array.flatten()
    
    
    
    
    
    
    
    
    
    
    
    
    # img = Image.open(image_path).convert('L')
    
    # # Apply threshold
    # if convert:
    #     pixel_array = np.array(img, dtype=np.uint8)
    #     binary_array = np.where(pixel_array >= threshold, 0.0, 1.0).astype(np.float32)
    #     return binary_array.flatten()
    # pixel_array = np.array(img)
    # return pixel_array.flatten()
