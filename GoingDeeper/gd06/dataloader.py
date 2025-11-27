import re
import six
import lmdb
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# Target characters
NUMBERS = "0123456789"
ENG_CHAR_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TARGET_CHARACTERS = ENG_CHAR_UPPER + NUMBERS


class LabelConverter(object):
    def __init__(self, character):
        # CTC는 blank를 내부적으로 처리하므로 character에 blank를 포함하지 않음
        self.character = character
        self.label_map = dict()
        for i, char in enumerate(self.character):
            # blank=0이므로 실제 문자는 1부터 시작
            self.label_map[char] = i + 1

    def encode(self, text):
        # CTC는 반복 문자를 자동으로 처리하므로 blank 삽입 불필요
        encoded_label = []
        for char in text:
            encoded_label.append(self.label_map[char])
        return np.array(encoded_label, dtype=np.int32)

    def decode(self, encoded_label):
        decoded_label = ""
        for encode in encoded_label:
            if encode > 0:  # blank(0) 무시
                decoded_label += self.character[encode - 1]
        return decoded_label


class MJDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        label_converter,
        img_size=(100, 32),
        max_text_len=22,
        character="",
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.label_converter = label_converter
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.character = character
        self.env = None

        # lmdb open to get num_samples only
        env = lmdb.open(dataset_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.num_samples = int(txn.get("num-samples".encode()))
            self.index_list = [idx + 1 for idx in range(self.num_samples)]
        env.close()

    def __len__(self):
        return self.num_samples

    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(self.dataset_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        return self.env

    def __getitem__(self, idx):
        index = self.index_list[idx]
        env = self._get_env()
        with env.begin(write=False) as txn:
            label_key = f"label-{index:09d}".encode()
            label = txn.get(label_key).decode("utf-8")

            img_key = f"image-{index:09d}".encode()
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img_pil = Image.open(buf).convert("RGB")
            except IOError:
                img_pil = Image.new("RGB", self.img_size)
                label = "-"

        orig_w, orig_h = img_pil.size
        target_width = min(int(orig_w * self.img_size[1] / orig_h), self.img_size[0])
        target_img_size = (target_width, self.img_size[1])
        img_pil = img_pil.resize(target_img_size)

        img = np.array(img_pil)
        img = img.transpose(2, 0, 1)

        padded_img = np.zeros((3, self.img_size[1], self.img_size[0]), dtype=np.float32)
        c, h, w = img.shape
        padded_img[:, :h, :w] = img

        # 레이블 전처리
        label = label.upper()
        out_of_char = f"[^{self.character}]"
        label = re.sub(out_of_char, "", label)
        label = label[: self.max_text_len]

        encoded_label = self.label_converter.encode(label)

        return padded_img, encoded_label, len(encoded_label), label


def collate_fn(batch):
    imgs, encoded_labels, label_lens, raw_labels = zip(*batch)

    # 이미지를 [0,1] 범위로 정규화
    imgs_normalized = []
    for img in imgs:
        imgs_normalized.append(img / 255.0)
    
    imgs_tensor = torch.tensor(np.stack(imgs_normalized, axis=0), dtype=torch.float32)

    max_len = max(label_lens)
    labels_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, label_arr in enumerate(encoded_labels):
        length = label_lens[i]
        labels_padded[i, :length] = torch.tensor(label_arr, dtype=torch.long)

    batch_size = imgs_tensor.size(0)
    # 모델 출력 시퀀스 길이는 35 (7×5)
    input_length = torch.full(size=(batch_size,), fill_value=35, dtype=torch.long)
    label_length = torch.tensor(label_lens, dtype=torch.long)

    return (
        imgs_tensor,
        labels_padded,
        input_length,
        label_length,
        raw_labels,  # 디버깅용
    )
