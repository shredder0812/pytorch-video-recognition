import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict

""" Dataset structure for endoscopy videos:
endoscopy_data/
  train/
    video1.mp4
    video1_label.txt
    video2.mp4
    video2_label.txt
  val/
    ...
  test/
    ...

"""

class EndoscopyVideoDataset(Dataset):
    r"""A Dataset for endoscopy videos where each video is a 16-frame sequence.
    Labels are stored in separate .txt files with suffix '_label.txt'.
    Returns triplets (anchor, positive, negative) for training C3D with TripletMarginLoss.

    Args:
        root_dir (str): Root directory containing split folders (train/val/test).
        split (str): 'train', 'val', or 'test'. Defaults to 'train'.
        clip_len (int): Number of frames in each clip. Defaults to 16.
    """

    def __init__(self, root_dir, split='train', clip_len=16):
        self.root_dir = root_dir
        self.split = split
        self.clip_len = clip_len
        self.crop_size = 112

        # Load video and label files
        self.videos, self.id_to_videos = self.load_videos_and_labels()
        print(f'Number of {split} videos: {len(self.videos)}')

    def load_videos_and_labels(self):
        """Load video files and group by track ID from label files."""
        folder = os.path.join(self.root_dir, self.split)
        if not os.path.exists(folder):
            raise RuntimeError(f"Folder {folder} not found.")

        videos = []  # List of (video_path, track_id)
        id_to_videos = defaultdict(list)  # {track_id: [video_paths]}

        for fname in sorted(os.listdir(folder)):
            if fname.endswith('.mp4'):  # Chỉ xử lý file video
                video_path = os.path.join(folder, fname)
                label_file = os.path.join(folder, f"{fname[:-4]}_label.txt")
                if not os.path.exists(label_file):
                    print(f"Warning: Label file {label_file} not found, skipping.")
                    continue
                
                with open(label_file, 'r') as f:
                    track_id = int(f.read().strip())  # Giả định file chứa 1 số nguyên là track_id
                
                videos.append((video_path, track_id))
                id_to_videos[track_id].append(video_path)

        return videos, id_to_videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        """Return a triplet (anchor, positive, negative) of clips."""
        # Get anchor clip
        anchor_path, anchor_id = self.videos[index]
        anchor = self.load_clip(anchor_path)

        # Get positive clip (from same track_id)
        positive_path = random.choice([p for p in self.id_to_videos[anchor_id] if p != anchor_path])
        if not positive_path:  # Nếu không có video khác, dùng lại anchor
            positive_path = anchor_path
        positive = self.load_clip(positive_path)

        # Get negative clip (from different track_id)
        negative_id = random.choice([id_ for id_ in self.id_to_videos.keys() if id_ != anchor_id])
        negative_path = random.choice(self.id_to_videos[negative_id])
        negative = self.load_clip(negative_path)

        # Apply augmentation and normalization
        if self.split == 'train':
            anchor = self.randomflip(anchor)
            positive = self.randomflip(positive)
            negative = self.randomflip(negative)

        anchor = self.normalize(anchor)
        positive = self.normalize(positive)
        negative = self.normalize(negative)

        return anchor, positive, negative

    def load_clip(self, video_path):
        """Load a clip of 16 frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count != self.clip_len:
            raise RuntimeError(f"Video {video_path} has {frame_count} frames, expected {self.clip_len}.")

        while len(frames) < self.clip_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.crop_size, self.crop_size))  # Resize toàn bộ frame
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        
        cap.release()
        if len(frames) < self.clip_len:
            frames.extend([frames[-1]] * (self.clip_len - len(frames)))  # Pad nếu thiếu
        
        return torch.stack(frames).unsqueeze(0)  # [1, 3, 16, 112, 112]

    def randomflip(self, buffer):
        """Horizontally flip the clip randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i in range(buffer.shape[1]):  # Iterate over frames
                buffer[:, i] = torch.flip(buffer[:, i], dims=[2])  # Flip horizontally
        return buffer

    def normalize(self, buffer):
        """Normalize the clip (optional, adjust based on pretrained C3D requirements)."""
        mean = torch.tensor([90.0, 98.0, 102.0]).view(3, 1, 1, 1)
        buffer -= mean  # Subtract mean
        return buffer

if __name__ == "__main__":
    # Example usage
    train_data = EndoscopyVideoDataset(
        root_dir='endoscopy_data',
        split='train',
        clip_len=16
    )
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True, num_workers=4)

    for i, (anchor, positive, negative) in enumerate(train_loader):
        print(f"Anchor size: {anchor.size()}")
        print(f"Positive size: {positive.size()}")
        print(f"Negative size: {negative.size()}")
        if i == 1:
            break