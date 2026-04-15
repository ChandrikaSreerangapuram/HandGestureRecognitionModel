import cv2

def test_cameras():
    # Try indices 0 to 5
    for i in range(5):
        print(f"Testing camera index {i}...")
        # Use CAP_DSHOW on Windows for faster/more reliable initialization
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"SUCCESS: Camera found at index {i}")
            ret, frame = cap.read()
            if ret:
                print(f"Captured frame successfully from index {i}")
                cv2.imshow(f'Camera {i}', frame)
                cv2.waitKey(2000) # Show for 2 seconds
            cap.release()
            cv2.destroyAllWindows()
            return i
        else:
            print(f"Failed to open camera at index {i}")
    return -1

if __name__ == "__main__":
    found_idx = test_cameras()
    if found_idx == -1:
        print("CRITICAL: No webcam detected. Please check your hardware connections or privacy settings.")
    else:
        print(f"Use index {found_idx} in your main script.")
