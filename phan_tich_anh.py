import cv2
import numpy as np
import os 

# ===================================================================
# ⚠️ CÔNG CỤ ĐẾM RỆP SÁP ⚠️
# ===================================================================

# 1. Đường dẫn tới file ảnh
image_path = 'img1.jpg'

# 2. Tọa độ và Bán kính của Vùng đếm
center_x = 641
center_y = 641
radius = 300 # Bán kính vùng đếm (ROI)

# 3. Ngưỡng màu của rệp sáp
lower_green = np.array([30, 0, 100])
upper_green = np.array([85, 255, 255])

# 4. Tùy chọn lưu ảnh
SAVE_PROCESSED_IMAGES = True
output_folder = "processed_images"

# ===================================================================
# ⚙️ CHỨC NĂNG XỬ LÝ ẢNH
# ===================================================================

def analyze_image_and_count_dots(image_path, center_x, center_y, radius, lower_color, upper_color, save_images=False, output_folder="processed_images"):

    # Đọc ảnh
    image = cv2.imread(image_path)

    if image is None:
        print(f"Lỗi: Không thể đọc được file ảnh tại {image_path}. Vui lòng kiểm tra đường dẫn.")
        return

    height, width, _ = image.shape
    
    # 1. Phân đoạn Màu (Segmentation)
    # Chuyển đổi sang HSV để phân biệt màu tốt hơn
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Tạo mask (mặt nạ) để chỉ giữ lại màu trong dải ngưỡng
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Lọc nhiễu nhỏ (tùy chọn):
    # color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel=np.ones((3,3),np.uint8))

    # 2. Tạo Vùng Đếm
    # Tạo một ảnh đen có cùng kích thước
    mask_roi = np.zeros((height, width), dtype=np.uint8)
    
    # Vẽ hình tròn trắng cho vùng đếm
    cv2.circle(mask_roi, (center_x, center_y), radius, 255, thickness=-1)

    # 3. Tính toán
 
    # Lấy các chấm màu CHỈ nằm trong ROI bằng phép toán logic AND
    dots_in_roi = cv2.bitwise_and(color_mask, color_mask, mask=mask_roi)

    # Đếm tổng số pixel trắng (chấm màu đã chọn)
    N_total = cv2.countNonZero(color_mask)    # Tổng số chấm màu trong TOÀN BỘ ảnh
    N_in = cv2.countNonZero(dots_in_roi)      # Tổng số chấm màu CHỈ trong ROI

    # 4. Tính Tỷ lệ %
    if N_total > 0:
        percentage = (N_in / N_total) * 100
    else:
        percentage = 0
        print("Cảnh báo: Không tìm thấy rệp sáp nào trong toàn bộ ảnh (N_total = 0).")

    # 5. Hiển thị Kết quả và Trực quan hóa
    
    # Tạo một bản sao của ảnh gốc để vẽ ROI lên đó mà không làm thay đổi ảnh gốc
    image_with_roi = image.copy()
    cv2.circle(image_with_roi, (center_x, center_y), radius, (0, 0, 255), 2) # Màu đỏ (BGR)

    print("\n--- KẾT QUẢ PHÂN TÍCH ---")
    print(f"1. Tổng số rệp sáp trong toàn bộ ảnh: {N_total}")
    print(f"2. Số pixel rệp sáp trong Vùng đếm: {N_in}")
    print(f"3. TỶ LỆ % Rệp sáp trong vùng đếm: {percentage:.2f}%")
    print("----------------------------")

    cv2.imshow('1. Anh Goc va ROI', image_with_roi)
    cv2.imshow('2. Color Mask (Chon mau)', color_mask)
    cv2.imshow('3. Dots in ROI (Ket qua cuoi)', dots_in_roi)
    
    # ===================================================================
    # 💾 LƯU ẢNH SAU XỬ LÝ
    # ===================================================================
    if save_images:
        # Tạo thư mục đầu ra nếu chưa tồn tại
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Lấy tên file gốc (không có phần mở rộng)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Lưu các ảnh
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_original_with_ROI.png"), image_with_roi)
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_color_mask.png"), color_mask)
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_dots_in_roi.png"), dots_in_roi)
        print(f"\nẢnh đã xử lý được lưu vào thư mục: {output_folder}")

    # Nhấn phím bất kỳ để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===================================================================
# 🚀 CHẠY CHƯƠNG TRÌNH
# ===================================================================

if __name__ == "__main__":
    analyze_image_and_count_dots(
        image_path, 
        center_x, 
        center_y, 
        radius, 
        lower_green, # Sử dụng ngưỡng màu
        upper_green, # Sử dụng ngưỡng màu
        save_images=SAVE_PROCESSED_IMAGES,
        output_folder=output_folder
    )