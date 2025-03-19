import os

def fix_obj_id_in_tracking(track_root):
    # Duyệt qua các thư mục trong vid_track
    for track_dir in os.listdir(track_root):
        track_path = os.path.join(track_root, track_dir)
        if os.path.isdir(track_path) and track_dir.startswith('tracking_'):
            # Tìm file mot_result.txt
            mot_result_file = os.path.join(track_path, 'mot_result.txt')
            if not os.path.exists(mot_result_file):
                print(f"mot_result.txt not found in {track_path}, skipping.")
                continue

            # Đọc nội dung mot_result.txt
            with open(mot_result_file, 'r') as f:
                lines = f.readlines()

            # Tạo nội dung mới với obj_id = 1
            fixed_lines = []
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:  # Đảm bảo có ít nhất frame_id và obj_id
                    parts[1] = '1'  # Thay obj_id thành 1
                    fixed_lines.append(','.join(parts))

            # Ghi vào mot_result_fix.txt
            mot_result_fix_file = os.path.join(track_path, 'mot_result_fix.txt')
            with open(mot_result_fix_file, 'w') as f:
                f.write('\n'.join(fixed_lines))
                f.write('\n')  # Thêm dòng trống cuối file nếu cần

            print(f"Created {mot_result_fix_file} with obj_id set to 1.")

if __name__ == "__main__":
    track_root = 'vid_track'  # Thư mục chứa các folder tracking
    fix_obj_id_in_tracking(track_root)