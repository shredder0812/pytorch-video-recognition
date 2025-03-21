import os
import cv2
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time

def process_tracking_data(vid_root, track_root, output_root, clip_len=16):
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'test'), exist_ok=True)

    # Xóa thư mục temp nếu tồn tại để tránh xung đột
    temp_dir = os.path.join(output_root, 'temp')
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    all_clips = []

    # Duyệt qua các thư mục tracking
    for track_dir in os.listdir(track_root):
        if not track_dir.startswith('tracking_'):
            continue

        # Xác định prefix và ID từ tên folder tracking
        prefix = track_dir.split('_')[1]  # UTDD hoặc UTTQ
        track_id = track_dir.split('_')[-1]  # Ví dụ: 26320BVK020
        vid_folder = os.path.join(vid_root, prefix)

        # Tìm video gốc gần đúng
        vid_path = None
        for vid_file in os.listdir(vid_folder):
            if vid_file.endswith('.mp4'):
                if track_id in vid_file or prefix in vid_file:
                    vid_path = os.path.join(vid_folder, vid_file)
                    print(f"Found potential match: {vid_path} for {track_dir}")
                    break
        if not vid_path:
            print(f"No matching video found for {track_dir} in {vid_folder}, skipping.")
            continue

        # Đọc file mot_result.txt
        track_file = os.path.join(track_root, track_dir, 'mot_result.txt')
        if not os.path.exists(track_file):
            print(f"Tracking file {track_file} not found, skipping.")
            continue

        tracks = defaultdict(list)
        with open(track_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                frame_id, obj_id, x1, y1, w, h, conf, _, _, _ = map(float, parts[:10])
                x2 = x1 + w
                y2 = y1 + h
                tracks[int(obj_id)].append((int(frame_id), int(x1), int(y1), int(x2), int(y2)))

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"Cannot open video {vid_path}, skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video {vid_path}: {frame_width}x{frame_height}, FPS: {fps}")

        # Sắp xếp các track theo obj_id và frame_id
        track_list = []
        for obj_id, frames in tracks.items():
            frames.sort(key=lambda x: x[0])  # Sắp xếp theo frame_id
            track_list.append((obj_id, frames))
        track_list.sort(key=lambda x: (x[0], x[1][0][0]))  # Sắp xếp theo obj_id và frame đầu tiên

        # Gộp các track không đủ 16 frame với track gần nhất trước đó
        merged_tracks = []
        current_track = None
        for obj_id, frames in track_list:
            if len(frames) < clip_len:
                print(f"Track {obj_id} has {len(frames)} frames, merging with previous track.")
                if current_track is None:
                    # Nếu không có track trước đó, lưu track hiện tại để gộp với track tiếp theo
                    current_track = (obj_id, frames)
                    continue
                # Gộp với track trước đó
                prev_obj_id, prev_frames = current_track
                prev_frames.extend(frames)
                print(f"Merged track {obj_id} into track {prev_obj_id}, new length: {len(prev_frames)}")
                current_track = (prev_obj_id, prev_frames)
            else:
                if current_track is not None:
                    merged_tracks.append(current_track)
                current_track = (obj_id, frames)

        # Thêm track cuối cùng (nếu có)
        if current_track is not None:
            merged_tracks.append(current_track)

        # Xử lý các track đã gộp để tạo clip
        for obj_id, frames in merged_tracks:
            frames.sort(key=lambda x: x[0])  # Sắp xếp lại theo frame_id
            print(f"Processing merged track {obj_id} with {len(frames)} frames.")

            # Chỉ tạo clip nếu track có đủ 16 frame
            if len(frames) < clip_len:
                print(f"Merged track {obj_id} still has only {len(frames)} frames, skipping.")
                continue

            for start_idx in range(0, len(frames) - clip_len + 1, clip_len):
                clip_frames = frames[start_idx:start_idx + clip_len]
                frame_ids = [f[0] for f in clip_frames]

                # Kiểm tra nếu clip không đủ frame liên tiếp hoặc có frame trùng lặp
                if len(set(frame_ids)) != clip_len:
                    print(f"Clip at start {frame_ids[0]} skipped due to duplicate frames.")
                    continue
                if max(frame_ids) >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    print(f"Clip at start {frame_ids[0]} skipped due to frame out of range.")
                    continue

                # Tạo tên file duy nhất (thêm timestamp để tránh trùng lặp)
                vid_name = os.path.basename(vid_path)[:-4]  # Ví dụ: 210504CS205
                unique_suffix = time.strftime("%H%M%S")
                clip_name = f"video_{prefix}_{vid_name}_track_{obj_id}_start_{frame_ids[0]}_{unique_suffix}"
                clip_path = os.path.join(output_root, 'temp', f"{clip_name}.mp4")
                label_path = os.path.join(output_root, 'temp', f"{clip_name}_label.txt")

                # Tạo video clip
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(clip_path, fourcc, fps, (112, 112))
                for frame_id, x1, y1, x2, y2 in clip_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Failed to read frame {frame_id} from {vid_path}.")
                        break
                    crop = frame[max(0, y1):min(frame_height, y2), max(0, x1):min(frame_width, x2)]
                    if crop.size == 0:
                        crop = np.zeros((112, 112, 3), dtype=np.uint8)
                    crop = cv2.resize(crop, (112, 112))
                    out.write(crop)
                out.release()

                # Lưu label (obj_id)
                with open(label_path, 'w') as f:
                    f.write(str(obj_id))

                all_clips.append((clip_path, label_path))

        cap.release()

    if not all_clips:
        print("No clips were generated. Check video files and tracking data.")
        return

    # Chia dữ liệu thành train/val/test
    train_clips, test_clips = train_test_split(all_clips, test_size=0.3, random_state=42)
    val_clips, test_clips = train_test_split(test_clips, test_size=0.5, random_state=42)

    # Di chuyển các clip vào thư mục tương ứng
    for split, clips in [('train', train_clips), ('val', val_clips), ('test', test_clips)]:
        for clip_path, label_path in clips:
            dest_clip_path = os.path.join(output_root, split, os.path.basename(clip_path))
            dest_label_path = os.path.join(output_root, split, os.path.basename(label_path))
            if os.path.exists(dest_clip_path):
                print(f"Destination {dest_clip_path} exists, overwriting.")
                os.remove(dest_clip_path)
            if os.path.exists(dest_label_path):
                print(f"Destination {dest_label_path} exists, overwriting.")
                os.remove(dest_label_path)
            os.rename(clip_path, dest_clip_path)
            os.rename(label_path, dest_label_path)

    print("Dataset creation completed.")
    print(f"Train clips: {len(train_clips)}, Val clips: {len(val_clips)}, Test clips: {len(test_clips)}")

if __name__ == "__main__":
    vid_root = 'vid_endocv'
    track_root = 'vid_track'
    output_root = 'endoc3d_data'
    process_tracking_data(vid_root, track_root, output_root, clip_len=16)