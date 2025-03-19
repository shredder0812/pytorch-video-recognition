import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict

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
        self.videos, self.name_to_videos = self.load_videos_and_labels()
        print(f'Number of {split} videos: {len(self.videos)}')

    def load_videos_and_labels(self):
        """Load video files and group by video name prefix (e.g., UTDD_230320BVK020)."""
        folder = os.path.join(self.root_dir, self.split)
        if not os.path.exists(folder):
            raise RuntimeError(f"Folder {folder} not found.")

        videos = []  # List of (video_path, track_id)
        name_to_videos = defaultdict(list)  # {video_name_prefix: [video_paths]}

        for fname in sorted(os.listdir(folder)):
            if fname.endswith('.mp4'):
                video_path = os.path.join(folder, fname)
                label_file = os.path.join(folder, f"{fname[:-4]}_label.txt")
                if not os.path.exists(label_file):
                    print(f"Warning: Label file {label_file} not found, skipping.")
                    continue
                
                with open(label_file, 'r') as f:
                    track_id = int(f.read().strip())
                
                # Trích xuất phần tên video (ví dụ: UTDD_230320BVK020)
                video_name_prefix = ''.join(os.path.basename(video_path).split('')[1:3])
                videos.append((video_path, track_id))
                name_to_videos[video_name_prefix].append(video_path)

        return videos, name_to_videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        """Return a triplet (anchor, positive, negative) of clips."""
        # Get anchor clip
        anchor_path, anchor_id = self.videos[index]
        anchor = self.load_clip(anchor_path)

        # Trích xuất video name prefix của anchor
        anchor_name_prefix = ''.join(os.path.basename(anchor_path).split('')[1:3])

        # Get positive clip (cùng video name prefix)
        positive_candidates = [p for p in self.name_to_videos[anchor_name_prefix] if p != anchor_path]
        positive_path = random.choice(positive_candidates) if positive_candidates else anchor_path
        positive = self.load_clip(positive_path)

        # Get negative clip (khác video name prefix)
        negative_name_prefixes = [name for name in self.name_to_videos.keys() if name != anchor_name_prefix]
        if not negative_name_prefixes:
            negative_path = anchor_path
        else:
            negative_name_prefix = random.choice(negative_name_prefixes)
            negative_path = random.choice(self.name_to_videos[negative_name_prefix])
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
            frame = cv2.resize(frame, (self.crop_size, self.crop_size))
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0  # [3, 112, 112]
            frames.append(frame)
        
        cap.release()
        if len(frames) < self.clip_len:
            frames.extend([frames[-1]] * (self.clip_len - len(frames)))
        
        clip = torch.stack(frames)  # [16, 3, 112, 112]
        clip = clip.permute(1, 0, 2, 3)  # [3, 16, 112, 112]
        return clip

    def randomflip(self, buffer):
        """Horizontally flip the clip randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i in range(buffer.shape[1]):  # Iterate over frames (dimension 1)
                buffer[:, i] = torch.flip(buffer[:, i], dims=[2])  # Flip horizontally (dimension 3)
        return buffer

    def normalize(self, buffer):
        """Normalize the clip."""
        mean = torch.tensor([90.0, 98.0, 102.0]).view(3, 1, 1, 1)
        if buffer.shape[0] != 3:  # Kiểm tra số kênh (chiều 0 sau khi bỏ batch dim)
            raise RuntimeError(f"Expected buffer to have 3 channels, got shape {buffer.shape}")
        buffer -= mean
        return buffer

if __name__ == "__main__":
    # Example usage
    train_data = EndoscopyVideoDataset(
        root_dir='endoc3d_data',
        split='train',
        clip_len=16
    )
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True, num_workers=2)

    for i, (anchor, positive, negative) in enumerate(train_loader):
        print(f"Anchor size: {anchor.size()}")
        print(f"Positive size: {positive.size()}")
        print(f"Negative size: {negative.size()}")
        if i == 1:
            break