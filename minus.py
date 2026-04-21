import cv2

def subtract_images(image_a_path, image_b_path, output_path):
    # 1. 讀取照片 A 與 照片 B
    img_a = cv2.imread(image_a_path)
    img_b = cv2.imread(image_b_path)

    # 檢查照片是否成功讀取
    if img_a is None or img_b is None:
        print("錯誤：無法讀取照片，請確認檔案路徑是否正確。")
        return

    # 2. 確保兩張照片大小一致 (以照片 A 的大小為基準來縮放照片 B)
    height, width = img_a.shape[:2]
    img_b_resized = cv2.resize(img_b, (width, height))

    # 3. 執行相減操作
    # 方法一：絕對差異 (推薦，常用於找出兩張圖片不同的地方)
    # result_img = cv2.absdiff(img_a, img_b_resized)
    
    # 方法二：單純相減 (如果你只要 A - B，把上面那行註解掉，改用下面這行)
    result_img = cv2.subtract(img_a, img_b_resized)

    # 4. 儲存結果與顯示
    cv2.imwrite(output_path, result_img)
    print(f"處理完成！結果已儲存至：{output_path}")

    # 顯示結果 (按任意鍵關閉視窗)
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================= 執行區塊 =================
if __name__ == "__main__":
    # 請將這裡替換成你實際的照片路徑
    photo_b = "/Users/corn/Downloads/output_Normal_Rail.jpg"  
    photo_a = "Output/20241126_Background.jpg"  
    output = "result_diff.jpg"

    subtract_images(photo_b, photo_a, output)