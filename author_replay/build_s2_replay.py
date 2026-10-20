#!/usr/bin/env python3
"""Build a memory-bounded replay of the author's S2_Parallel_Workflow/s2_u_net_tf.py.

Unmodified, the script needs more than 16 GB (float64 images, int64 and float32 one-hot
masks, plus Keras' own copies) and was killed by the 16 GB worker limit. This copy keeps
the author's model, metrics, compile, MirroredStrategy, 80/20 split (random_state=0),
batch size 32, 50 epochs and shuffle=False verbatim, and replaces only the data handling:
tiles and encoded masks stay uint8 in memory and each batch of 32 is normalised
(keras normalize, axis=1, per image as before) and one-hot encoded on the fly, in the same
order. The LabelEncoder mapping is reproduced exactly (sorted unique mask values -> 0..k-1).
Remaining difference: images reach the model as float32 (TensorFlow casts the float64
array to float32 in the original as well).
"""
import sys

src = open(sys.argv[1] if len(sys.argv) > 1 else "../../S2_Parallel_Workflow/s2_u_net_tf.py").read()

START = "train_images = []"
END = 'print(y_train_cat.shape, " y shape and len ", len(y_train_cat))'
i, j = src.index(START), src.index(END) + len(END)
assert src.count(START) == 1 and src.count(END) == 1

DATA = '''train_images = []
for directory_path in glob.glob("train_images_4032/"):
    for img_path in sorted(glob.glob(os.path.join(directory_path, "*.png"))):
        train_images.append(cv2.imread(img_path, 0))
train_images = np.expand_dims(np.array(train_images, dtype=np.uint8), axis=3)
train_masks = []
for directory_path in glob.glob("train_masks_4032/"):
    for mask_path in sorted(glob.glob(os.path.join(directory_path, "*.png"))):
        train_masks.append(cv2.imread(mask_path, 0))
train_masks = np.array(train_masks, dtype=np.uint8)

# LabelEncoder.fit_transform, reproduced: sorted unique values -> 0..k-1 (uint8 lookup)
_classes = np.unique(train_masks)
_lut = np.zeros(256, dtype=np.uint8); _lut[_classes] = np.arange(len(_classes), dtype=np.uint8)
train_masks_input = np.expand_dims(_lut[train_masks], axis=3)
del train_masks
print("REPLAY label classes (mask gray -> class):", dict(zip(_classes.tolist(), range(len(_classes)))))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.20, random_state = 0)
del train_images, train_masks_input
print("Class values in the dataset are ... ", np.unique(y_train))

from keras.utils import to_categorical

class _Batches(tf.keras.utils.Sequence):
    """Batches of 32 in index order: normalize(axis=1) and one-hot per batch."""
    def __init__(self, x, y, batch=32):
        self.x, self.y, self.batch = x, y, batch
    def __len__(self):
        return (len(self.x) + self.batch - 1) // self.batch
    def __getitem__(self, k):
        s = slice(k * self.batch, (k + 1) * self.batch)
        xb = normalize(self.x[s].astype(np.float64), axis=1).astype(np.float32)
        yb = self.y[s]
        yb = to_categorical(yb, num_classes=n_classes).reshape(
            (yb.shape[0], yb.shape[1], yb.shape[2], n_classes)).astype(np.float32)
        return xb, yb

train_seq = _Batches(X_train, y_train)
test_seq = _Batches(X_test, y_test)
print((len(X_train), X_train.shape[1], X_train.shape[2], n_classes), " y shape and len ", len(X_train))'''

out = src[:i] + DATA + src[j:]
for old, new in [
    ("history = model.fit(X_train, y_train_cat,", "history = model.fit(train_seq,"),
    ("batch_size = 32, #BATCH_SIZE,#1,#32,", "#batch_size = 32 (set by train_seq),"),
    ("validation_data=(X_test, y_test_cat),", "validation_data=test_seq,"),
    ("f1_score, precision, recall = model.evaluate(X_test, y_test_cat, verbose=1)",
     "f1_score, precision, recall = model.evaluate(test_seq, verbose=1)"),
]:
    assert out.count(old) == 1, old
    out = out.replace(old, new)
hdr = ("print('REPLAY of the author s2_u_net_tf.py with batched data handling (model, metrics, "
       "compile, split, batch 32, 50 epochs, shuffle=False unchanged); see build_s2_replay.py')\n")
open("s2_u_net_tf_replay.py", "w").write(hdr + out)
print("wrote s2_u_net_tf_replay.py")
