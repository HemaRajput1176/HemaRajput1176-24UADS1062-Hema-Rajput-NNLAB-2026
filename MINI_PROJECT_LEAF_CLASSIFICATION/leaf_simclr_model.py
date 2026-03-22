!nvidia-smi
!pip install torch torchvision scikit-learn matplotlib seaborn Pillow tqdm numpy -q

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\n  Device  : {DEVICE}')
if DEVICE == 'cuda':
    print(f'  GPU     : {torch.cuda.get_device_name(0)}')
    print(f'  Memory  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print('\n  GPU ready — proceed to Cell 2')
else:
    print('\n  No GPU detected!')
    print('  Go to: Runtime → Change runtime type → T4 GPU → Save')
    print('  Then run this cell again')
print(f'  PyTorch : {torch.__version__}')
from google.colab import drive
import zipfile, os
from pathlib import Path

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)
print('Drive mounted')

# Search for zip file in Drive
print('\nSearching for dataset zip...')
zip_path = None
search_paths = [
    '/content/drive/MyDrive/LeafProject/leaf_small.zip',
    '/content/drive/MyDrive/leaf_small.zip',
    '/content/drive/MyDrive/LeafProject/Leave-Small.zip',
]
for p in search_paths:
    if Path(p).exists():
        zip_path = p; break

# If not found in common paths, search all Drive
if not zip_path:
    for root, dirs, files in os.walk('/content/drive/MyDrive'):
        for f in files:
            if f.endswith('.zip'):
                full = os.path.join(root, f)
                size = os.path.getsize(full)/1024/1024
                print(f'  Found: {f}  ({size:.1f} MB)  in {root}')
                if zip_path is None:
                    zip_path = full

if not zip_path:
    print('  No zip found. Upload leaf_small.zip to Google Drive → LeafProject folder')
else:
    size_mb = os.path.getsize(zip_path)/1024/1024
    print(f'  Using: {zip_path}  ({size_mb:.1f} MB)')
    print('  Extracting...')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall('/content/')
    print('  Extracted!')

    # Auto-find dataset path
    RAW_FOLDER = None
    for root, dirs, files_list in os.walk('/content'):
        if 'drive' in root: continue
        jpgs    = [f for f in files_list if f.lower().endswith('.jpg')]
        sp_dirs = [d for d in Path(root).iterdir() if d.is_dir()]
        if len(jpgs) >= 30 and len(sp_dirs) >= 3:
            RAW_FOLDER = root; break

    if not RAW_FOLDER and Path('/content/Leave-Small').exists():
        RAW_FOLDER = '/content/Leave-Small'

    if RAW_FOLDER:
        species = [d.name for d in Path(RAW_FOLDER).iterdir() if d.is_dir()]
        total   = sum(len(list(Path(RAW_FOLDER,sp).glob('*.jpg'))) for sp in species)
        print(f'\n  Dataset found!')
        print(f'  Path    : {RAW_FOLDER}')
        print(f'  Species : {species}')
        print(f'  Images  : {total}')
        print('\n  Proceed to Cell 3')
    else:
        RAW_FOLDER = '/content/Leave-Small'
        print(f'  RAW_FOLDER set to: {RAW_FOLDER}')
        print('  Contents of /content/:')
        for item in os.listdir('/content'):
            print(f'    {item}')
          %matplotlib inline
import os, shutil, random, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import (
    Dataset, DataLoader, random_split, Subset
)
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 110
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  Device: {DEVICE}')
if DEVICE != 'cuda':
    print('  Switch to T4 GPU — Runtime → Change runtime type → T4 GPU')
  # ══════════════════════════════════════════════
#  PATHS
# RAW_FOLDER is set automatically in Cell 2
# If needed set manually: RAW_FOLDER = '/content/Leave-Small'
# ══════════════════════════════════════════════
AUGMENT_FOLDER = '/content/augmented_dataset'
CHECKPOINT_DIR = '/content/checkpoints'
OUTPUT_DIR     = '/content/outputs'

# ══════════════════════════════════════════════
#  TRAINING SETTINGS
# ══════════════════════════════════════════════
IMAGE_SIZE       = 224
BATCH_SIZE       = 64
TARGET_TOTAL     = 4000   # augment dataset to exactly 4000 images
SSL_EPOCHS       = 100
FT_EPOCHS        = 30
TEMPERATURE      = 0.5
PROJECTION_DIM   = 128
SSL_LR           = 3e-4
FT_LR            = 1e-3
NUM_WORKERS      = 2

SSL_CKPT = os.path.join(CHECKPOINT_DIR, 'ssl_encoder.pth')
FT_CKPT  = os.path.join(CHECKPOINT_DIR, 'finetuned.pth')

for d in [AUGMENT_FOLDER, CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# Verify dataset
if Path(RAW_FOLDER).exists():
    species = [d.name for d in Path(RAW_FOLDER).iterdir() if d.is_dir()]
    raw_total = sum(len(list(Path(RAW_FOLDER,sp).glob('*.jpg'))) for sp in species)
    print(f'  Config ready')
    print(f'  Species ({len(species)}) : {species}')
    print(f'  Raw images   : {raw_total}')
    print(f'  Target total : {TARGET_TOTAL} images after augmentation')
    print(f'  Batch size   : {BATCH_SIZE}')
    print(f'  SSL epochs   : {SSL_EPOCHS}')
    print(f'  Device       : {DEVICE}')
else:
    print(f'  RAW_FOLDER not found: {RAW_FOLDER}')
    print('  Re-run Cell 2 or set RAW_FOLDER manually')
  print('='*55)
print('  STEP 1 — Load Raw Dataset')
print('='*55)

raw_species   = {}
all_img_paths = []

for sp_dir in sorted(Path(RAW_FOLDER).iterdir()):
    if not sp_dir.is_dir(): continue
    imgs = list(sp_dir.glob('*.jpg')) + list(sp_dir.glob('*.JPG'))
    if not imgs: continue
    raw_species[sp_dir.name] = imgs
    all_img_paths.extend(imgs)
    print(f'  {sp_dir.name:<30} {len(imgs):>4} images')

print(f'\n  Total species : {len(raw_species)}')
print(f'  Total images  : {len(all_img_paths)}')

# Distribution chart
plt.figure(figsize=(13, 4))
bars = plt.bar(raw_species.keys(),
               [len(v) for v in raw_species.values()],
               color='#E87C4C', edgecolor='white')
for b in bars:
    plt.text(b.get_x()+b.get_width()/2, b.get_height()+1,
             str(int(b.get_height())), ha='center', fontsize=9)
plt.title(f'Raw Dataset — {len(all_img_paths)} images, {len(raw_species)} species')
plt.ylabel('Images'); plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'raw_distribution.png'), dpi=150)
plt.show()

# Sample images grid
n_sp = len(raw_species)
fig, axes = plt.subplots(n_sp, 3, figsize=(9, n_sp*2.5))
for row,(sp_name,img_list) in enumerate(raw_species.items()):
    samples = random.sample(img_list, min(3, len(img_list)))
    for col in range(3):
        ax = axes[row][col] if n_sp>1 else axes[col]
        if col<len(samples):
            ax.imshow(Image.open(samples[col]).convert('L'), cmap='gray')
            if col==0:
                ax.set_ylabel(sp_name, fontsize=8, rotation=0,
                              ha='right', va='center', labelpad=60)
        ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('Sample Leaf Images — MBMU Campus (Grayscale)', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'sample_images.png'), dpi=150, bbox_inches='tight')
plt.show()
print('  Charts saved')
print('='*55)
print('  STEP 2 — Augment Dataset')
print(f'  Target: {TARGET_TOTAL} total images')
print('='*55)

# Skip if already done
if Path(AUGMENT_FOLDER).exists():
    existing = sum(len(list(sp.glob('*.jpg')))
                   for sp in Path(AUGMENT_FOLDER).iterdir() if sp.is_dir())
    if existing >= TARGET_TOTAL * 0.9:  # within 10% of target
        print(f'  Already exists: {existing:,} images — skipping')
        SKIP_AUG = True
    else:
        shutil.rmtree(AUGMENT_FOLDER)
        os.makedirs(AUGMENT_FOLDER, exist_ok=True)
        SKIP_AUG = False
else:
    SKIP_AUG = False

if not SKIP_AUG:
    # Calculate per-species target
    n_species        = len(raw_species)
    per_species_tgt  = TARGET_TOTAL // n_species   # e.g. 4000 // 10 = 400
    raw_per_species  = len(all_img_paths) // n_species
    augments_needed  = max(1, per_species_tgt // raw_per_species - 1)

    print(f'  Species        : {n_species}')
    print(f'  Per species    : {per_species_tgt} images')
    print(f'  Augments/image : {augments_needed}')
    print(f'  Expected total : ~{n_species * per_species_tgt:,} images\n')

    # Augmentation transforms
    AUG_POOL = [
        T.RandomHorizontalFlip(p=1.0),
        T.RandomVerticalFlip(p=1.0),
        T.RandomRotation(degrees=30),
        T.RandomRotation(degrees=90),
        T.RandomRotation(degrees=180),
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        T.RandomAffine(15, translate=(0.1,0.1), shear=10),
        T.CenterCrop(int(IMAGE_SIZE*0.85)),
        T.RandomAutocontrast(p=1.0),
        T.RandomEqualize(p=1.0),
        T.RandomAdjustSharpness(2.5, p=1.0),
        T.RandomAdjustSharpness(0.0, p=1.0),
        T.GaussianBlur(5, (0.5, 2.0)),
        T.RandomPosterize(4, p=1.0),
        T.RandomSolarize(128, p=1.0),
    ]

    def augment_one(img_pil, n):
        base = T.Compose([T.Grayscale(1), T.Resize((IMAGE_SIZE, IMAGE_SIZE))])
        img  = base(img_pil)
        out  = []
        for _ in range(n):
            k     = np.random.randint(2, 5)
            chain = T.Compose([AUG_POOL[j]
                               for j in np.random.choice(len(AUG_POOL), k, replace=False)])
            try:    aug = T.Resize((IMAGE_SIZE, IMAGE_SIZE))(chain(img))
            except: aug = img
            out.append(aug)
        return out

    print(f'  {"Species":<30} {"Orig":>6}  {"Total":>8}')
    print(f'  {"-"*48}')
    total_orig = total_aug = 0

    for sp_dir in sorted(Path(RAW_FOLDER).iterdir()):
        if not sp_dir.is_dir(): continue
        sp_clean = (sp_dir.name.replace('(','').replace(')','')
                              .replace(' ','_').strip())
        out_dir  = Path(AUGMENT_FOLDER) / sp_clean
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs   = list(sp_dir.glob('*.jpg')) + list(sp_dir.glob('*.JPG'))
        sp_tot = 0

        for img_path in imgs:
            # Copy original
            shutil.copy2(img_path, out_dir / img_path.name)
            sp_tot += 1

            # Stop if we reached per-species target
            if sp_tot >= per_species_tgt:
                break

            # Create augmented copies
            try:
                remaining = per_species_tgt - sp_tot
                n_aug     = min(augments_needed, remaining)
                augs      = augment_one(Image.open(img_path), n_aug)
                for j, aug in enumerate(augs):
                    if sp_tot >= per_species_tgt: break
                    aug.save(
                        str(out_dir / f'{img_path.stem}_aug{j+1:03d}.jpg'),
                        'JPEG', quality=92
                    )
                    sp_tot += 1
            except: pass

        total_orig += len(imgs)
        total_aug  += sp_tot
        print(f'  {sp_clean:<30} {len(imgs):>6}  {sp_tot:>8}')

    print(f'  {"-"*48}')
    print(f'  {"TOTAL":<30} {total_orig:>6}  {total_aug:>8}')
    print(f'\n  Augmentation complete ({total_aug:,} images)')
  # Encoder: ResNet-18 modified for grayscale
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base          = models.resnet18(weights=None)
        base.conv1    = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.feat_dim = base.fc.in_features  # 512
        base.fc       = nn.Identity()
        self.backbone = base
    def forward(self, x): return self.backbone(x)


# SimCLR: Encoder + projection head
class SimCLR(nn.Module):
    def __init__(self, encoder, proj_dim=128):
        super().__init__()
        self.encoder   = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )
    def forward(self, x):
        return nn.functional.normalize(
            self.projector(self.encoder(x)), dim=1
        )


# NT-Xent Loss — contrastive loss on embeddings NOT pixel reconstruction
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temp = temperature
        self.ce   = nn.CrossEntropyLoss()
    def forward(self, z1, z2):
        B   = z1.size(0)
        z   = torch.cat([z1, z2])
        sim = torch.mm(z, z.T) / self.temp
        sim.masked_fill_(
            torch.eye(2*B, device=z.device).bool(), float('-inf')
        )
        lbl = torch.cat([
            torch.arange(B, 2*B, device=z.device),
            torch.arange(0,   B, device=z.device)
        ])
        return self.ce(sim, lbl)


# LeafClassifier: frozen encoder + trainable head
class LeafClassifier(nn.Module):
    def __init__(self, encoder, num_classes, freeze=True):
        super().__init__()
        self.encoder = encoder
        if freeze:
            for p in encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(encoder.feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.head(self.encoder(x))


# Verify
_enc    = Encoder()
_mdl    = SimCLR(_enc, PROJECTION_DIM)
_x      = torch.zeros(2, 1, IMAGE_SIZE, IMAGE_SIZE)
_z      = _mdl(_x)
print('  Models defined')
print(f'  Encoder output    : 512-dim')
print(f'  Projection output : {_z.shape[1]}-dim')
print(f'  Parameters        : {sum(p.numel() for p in _mdl.parameters())/1e6:.2f}M')
print(f'  Loss type         : NT-Xent contrastive (NOT pixel reconstruction)')
# Two-view dataset for SimCLR
class TwoViewDataset(Dataset):
    def __init__(self, root, sz):
        self.ds  = ImageFolder(root)
        self.aug = T.Compose([
            T.Grayscale(1),
            T.RandomResizedCrop(sz, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.3),
            T.RandomRotation(30),
            T.RandomApply([T.GaussianBlur(9, (0.1, 2.0))], p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4)], p=0.8),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img, _ = self.ds[idx]                   # label ignored
        return self.aug(img), self.aug(img)      # two different views


ssl_ds  = TwoViewDataset(AUGMENT_FOLDER, IMAGE_SIZE)
ssl_ldr = DataLoader(
    ssl_ds, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS,
    pin_memory=True, drop_last=True
)
print(f'  SSL Dataset   : {len(ssl_ds):,} images')
print(f'  Batches/epoch : {len(ssl_ldr)}')
print(f'  Labels used   : NONE — self-supervised')
print(f'  Loss          : NT-Xent contrastive (NOT pixel reconstruction)\n')

# Build model
encoder   = Encoder()
ssl_model = SimCLR(encoder, PROJECTION_DIM).to(DEVICE)
criterion = NTXentLoss(TEMPERATURE)
optimizer = optim.Adam(ssl_model.parameters(), lr=SSL_LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SSL_EPOCHS)

ssl_losses  = []; best_loss = float('inf')
START_EPOCH = 1

# Resume from checkpoint if exists
if Path(SSL_CKPT).exists():
    ckpt = torch.load(SSL_CKPT, map_location=DEVICE)
    ssl_model.load_state_dict(ckpt['model_state'])
    START_EPOCH = ckpt['epoch'] + 1
    best_loss   = ckpt['loss']
    print(f'  Resumed from epoch {ckpt["epoch"]}  loss={ckpt["loss"]:.4f}')

print(f'  Starting SSL Pretraining — {SSL_EPOCHS} epochs on {DEVICE}\n')
print(f'  {"Epoch":<8} {"Loss":<12} {"LR":<14} {"Time"}')
print(f'  {"-"*46}')

for epoch in range(START_EPOCH, SSL_EPOCHS + 1):
    ssl_model.train(); t0 = time.time(); ep_loss = 0.0

    for v1, v2 in ssl_ldr:
        v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
        loss   = criterion(ssl_model(v1), ssl_model(v2))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        ep_loss += loss.item()

    avg = ep_loss / len(ssl_ldr)
    ssl_losses.append(avg); scheduler.step()

    if epoch % 10 == 0 or epoch == START_EPOCH:
        print(f'  {epoch:<8} {avg:<12.4f} '
              f'{scheduler.get_last_lr()[0]:<14.6f} {time.time()-t0:.0f}s')

    if avg < best_loss:
        best_loss = avg
        torch.save({
            'model_state': ssl_model.state_dict(),
            'epoch': epoch, 'loss': avg
        }, SSL_CKPT)

print(f'\n  Pretraining complete | Best loss: {best_loss:.4f}')

# Loss curve
plt.figure(figsize=(10, 4))
plt.plot(ssl_losses, linewidth=2, color='#7F77DD')
plt.fill_between(range(len(ssl_losses)), ssl_losses,
                 alpha=0.15, color='#7F77DD')
plt.title('SimCLR — NT-Xent Contrastive Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'ssl_loss.png'), dpi=150)
plt.show()
# Transforms
train_tf = T.Compose([
    T.Grayscale(1), T.Resize((IMAGE_SIZE+20, IMAGE_SIZE+20)),
    T.RandomCrop(IMAGE_SIZE), T.RandomHorizontalFlip(),
    T.RandomRotation(15), T.ToTensor(), T.Normalize([0.5],[0.5])
])
eval_tf = T.Compose([
    T.Grayscale(1), T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(), T.Normalize([0.5],[0.5])
])

# Dataset split: 70% train | 15% val | 15% test
full_tr = ImageFolder(AUGMENT_FOLDER, transform=train_tf)
full_ev = ImageFolder(AUGMENT_FOLDER, transform=eval_tf)
n       = len(full_tr)
n_tr    = int(n*0.70); n_vl = int(n*0.15); n_ts = n-n_tr-n_vl
g       = torch.Generator().manual_seed(SEED)
tr_ds, vl_ds, _  = random_split(full_tr, [n_tr, n_vl, n_ts], generator=g)
_, _,    ts_ds   = random_split(full_ev, [n_tr, n_vl, n_ts], generator=g)

kw     = dict(num_workers=NUM_WORKERS, pin_memory=True)
tr_ldr = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  **kw)
vl_ldr = DataLoader(vl_ds, BATCH_SIZE, shuffle=False, **kw)
ts_ldr = DataLoader(ts_ds, BATCH_SIZE, shuffle=False, **kw)
CLASS_NAMES = full_tr.classes; NUM_CLASSES = len(CLASS_NAMES)
print(f'  Train:{len(tr_ds):,}  Val:{n_vl:,}  Test:{n_ts:,}')
print(f'  Classes ({NUM_CLASSES}): {CLASS_NAMES}\n')

# Load SSL pretrained encoder
enc     = Encoder()
raw_wts = torch.load(SSL_CKPT, map_location=DEVICE)['model_state']
enc_wts = {k.replace('encoder.',''): v
           for k,v in raw_wts.items() if k.startswith('encoder.')}
enc.load_state_dict(enc_wts); enc = enc.to(DEVICE)
ft_model  = LeafClassifier(enc, NUM_CLASSES, freeze=True).to(DEVICE)
trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
print(f'  SSL encoder loaded | Trainable: {trainable:,} (encoder frozen)')

# Loss and optimizer
ce_loss  = nn.CrossEntropyLoss(label_smoothing=0.1)
ft_optim = optim.Adam(ft_model.head.parameters(), lr=FT_LR, weight_decay=1e-4)
ft_sched = optim.lr_scheduler.CosineAnnealingLR(ft_optim, T_max=FT_EPOCHS)

tr_losses,vl_losses,tr_accs,vl_accs = [],[],[],[]
best_val = 0.0

print(f'\n  {"Epoch":<8} {"Tr Loss":<10} {"Tr Acc":<10} {"Val Loss":<10} {"Val Acc"}')
print(f'  {"-"*52}')

for epoch in range(1, FT_EPOCHS+1):
    # Train
    ft_model.train(); tl=tc=tt=0
    for imgs,lbls in tr_ldr:
        imgs,lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        out  = ft_model(imgs); loss = ce_loss(out, lbls)
        ft_optim.zero_grad(); loss.backward(); ft_optim.step()
        tl += loss.item()
        tc += (out.argmax(1)==lbls).sum().item()
        tt += lbls.size(0)

    # Validate
    ft_model.eval(); vl=vc=vt=0
    with torch.no_grad():
        for imgs,lbls in vl_ldr:
            imgs,lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = ft_model(imgs)
            vl += ce_loss(out,lbls).item()
            vc += (out.argmax(1)==lbls).sum().item()
            vt += lbls.size(0)

    t_loss=tl/len(tr_ldr); v_loss=vl/len(vl_ldr)
    t_acc=tc/tt;            v_acc=vc/vt
    tr_losses.append(t_loss); vl_losses.append(v_loss)
    tr_accs.append(t_acc);   vl_accs.append(v_acc)
    ft_sched.step()

    if epoch%5==0 or epoch==1:
        print(f'  {epoch:<8} {t_loss:<10.4f} {t_acc:<10.3f} {v_loss:<10.4f} {v_acc:.3f}')

    if v_acc > best_val:
        best_val = v_acc
        torch.save({
            'model_state': ft_model.state_dict(),
            'class_names': CLASS_NAMES
        }, FT_CKPT)

print(f'\n  Fine-tuning done | Best val: {best_val:.4f} ({best_val*100:.2f}%)')
def quick_eval(enc_wts, train_b, test_b, n_cls,
               epochs=20, freeze=True, lr=1e-3):
    enc = Encoder(); enc.load_state_dict(enc_wts)
    mdl = LeafClassifier(enc, n_cls, freeze=freeze).to(DEVICE)
    opt = optim.Adam(
        mdl.head.parameters() if freeze else mdl.parameters(), lr=lr
    )
    ce = nn.CrossEntropyLoss(); mdl.train()
    for _ in range(epochs):
        for imgs,lbls in train_b:
            imgs,lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            loss = ce(mdl(imgs), lbls)
            opt.zero_grad(); loss.backward(); opt.step()
    mdl.eval(); correct=total=0; all_p=[]; all_l=[]
    with torch.no_grad():
        for imgs,lbls in test_b:
            imgs,lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            p = mdl(imgs).argmax(1)
            correct += (p==lbls).sum().item(); total += lbls.size(0)
            all_p.extend(p.cpu().tolist()); all_l.extend(lbls.cpu().tolist())
    return correct/total, all_p, all_l


# Load weights
raw_ssl     = torch.load(SSL_CKPT, map_location=DEVICE)['model_state']
ssl_enc_wts = {k.replace('encoder.',''): v
               for k,v in raw_ssl.items() if k.startswith('encoder.')}
scratch_wts = Encoder().state_dict()

eval_ds = ImageFolder(AUGMENT_FOLDER, transform=eval_tf)
n2      = len(eval_ds); n_test2 = max(int(n2*0.2), 50)
pool_ds, test_ds2 = random_split(
    eval_ds, [n2-n_test2, n_test2],
    generator=torch.Generator().manual_seed(SEED)
)
test_b2 = list(DataLoader(test_ds2, batch_size=64,
                           shuffle=False, num_workers=NUM_WORKERS))

# Experiment A: Few-Shot
print('Experiment A — Few-Shot Transfer\n')
print(f'  {"Shots":<10} {"SSL":<12} {"Scratch":<12} {"Gain"}')
print(f'  {"-"*44}')
shots=[1,5,10,20]; ssl_accs=[]; scr_accs=[]

for n_shot in shots:
    random.seed(SEED); cidx={}
    for i,(_,l) in enumerate(pool_ds): cidx.setdefault(l,[]).append(i)
    sel=[]
    for l,idxs in cidx.items(): sel.extend(random.sample(idxs,min(n_shot,len(idxs))))
    tb = list(DataLoader(Subset(pool_ds,sel), batch_size=min(32,len(sel)),
                         shuffle=True, num_workers=0))
    a_ssl,_,_ = quick_eval(ssl_enc_wts,tb,test_b2,NUM_CLASSES,epochs=20,freeze=True)
    a_scr,_,_ = quick_eval(scratch_wts, tb,test_b2,NUM_CLASSES,epochs=20,freeze=False)
    ssl_accs.append(a_ssl); scr_accs.append(a_scr)
    print(f'  {n_shot:<10} {a_ssl:<12.3f} {a_scr:<12.3f} '
          f'{a_ssl-a_scr:+.3f} {"✅" if a_ssl>a_scr else "⚠️"}')

x=np.arange(len(shots)); fig,ax=plt.subplots(figsize=(10,5))
b1=ax.bar(x-0.2,[a*100 for a in ssl_accs],0.4,label='SSL pretrained',color='#4C9BE8')
b2=ax.bar(x+0.2,[a*100 for a in scr_accs],0.4,label='From scratch',  color='#E87C4C')
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f'{b.get_height():.1f}%', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f'{s} shot/class' for s in shots])
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0,115)
ax.set_title('Exp A — Few-Shot: SSL vs Scratch (MBMU Leaf Species)')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'exp_a_fewshot.png'),dpi=150); plt.show()

# Experiment B: Unseen Species
print('\nExperiment B — Unseen Species Transfer\n')
random.seed(SEED); unseen_cls=random.sample(CLASS_NAMES,2)
seen_cls=[c for c in CLASS_NAMES if c not in unseen_cls]
print(f'  Seen   ({len(seen_cls)}) : {seen_cls}')
print(f'  Unseen (2)  : {unseen_cls}\n')
unseen_ids=[eval_ds.class_to_idx[c] for c in unseen_cls]
u_idxs=[i for i,(_,l) in enumerate(eval_ds) if l in unseen_ids]
u_sub=Subset(eval_ds,u_idxs); n_utr=int(len(u_sub)*0.5)
u_tr,u_ts=random_split(u_sub,[n_utr,len(u_sub)-n_utr],
                        generator=torch.Generator().manual_seed(SEED))
lbl_map={old:new for new,old in enumerate(unseen_ids)}
def remap(ds):
    return [(imgs, torch.tensor([lbl_map[l.item()] for l in lbls]))
            for imgs,lbls in DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)]
tr_b=remap(u_tr); ts_b=remap(u_ts)
a_ssl,p_ssl,l_ssl=quick_eval(ssl_enc_wts,tr_b,ts_b,2,epochs=30,freeze=True)
a_scr,_,_        =quick_eval(scratch_wts, tr_b,ts_b,2,epochs=30,freeze=False)
print(f'  SSL      : {a_ssl:.4f} ({a_ssl*100:.2f}%)')
print(f'  Scratch  : {a_scr:.4f} ({a_scr*100:.2f}%)')
print(f'  Gain     : {a_ssl-a_scr:+.4f}\n')
print(classification_report(l_ssl,p_ssl,target_names=unseen_cls,digits=3))
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(l_ssl,p_ssl),annot=True,fmt='d',cmap='Greens',
            xticklabels=unseen_cls,yticklabels=unseen_cls)
plt.title('Exp B — Unseen Species')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'exp_b_unseen.png'),dpi=150); plt.show()
print('  Transfer learning complete')
# ── Restore all variables lost after Colab restart ───────────
import os
from pathlib import Path

# Restore grand_total and aug_counts from augmented folder
aug_counts  = {}
grand_total = 0
for sp in sorted(Path(AUGMENT_FOLDER).iterdir()):
    if not sp.is_dir(): continue
    n = len(list(sp.glob('*.jpg')) + list(sp.glob('*.JPG')))
    aug_counts[sp.name] = n
    grand_total        += n

CLASS_NAMES = sorted(aug_counts.keys())
NUM_CLASSES = len(CLASS_NAMES)

# Restore all_img_paths from raw folder
raw_species   = {}
all_img_paths = []
for sp_dir in sorted(Path(RAW_FOLDER).iterdir()):
    if not sp_dir.is_dir(): continue
    imgs = list(sp_dir.glob('*.jpg')) + list(sp_dir.glob('*.JPG'))
    if not imgs: continue
    raw_species[sp_dir.name] = imgs
    all_img_paths.extend(imgs)

print('Variables restored:')
print(f'  grand_total   : {grand_total:,}')
print(f'  all_img_paths : {len(all_img_paths):,}')
print(f'  CLASS_NAMES   : {CLASS_NAMES}')
print(f'  NUM_CLASSES   : {NUM_CLASSES}')
from google.colab import files
import zipfile

zip_path = '/content/mini_project_1_results.zip'
with zipfile.ZipFile(zip_path, 'w') as z:
    for f in os.listdir(OUTPUT_DIR):
        z.write(os.path.join(OUTPUT_DIR,f), f'outputs/{f}')
    for f in os.listdir(CHECKPOINT_DIR):
        z.write(os.path.join(CHECKPOINT_DIR,f), f'checkpoints/{f}')

print('Files included:')
for f in os.listdir(OUTPUT_DIR):
    print(f'  outputs/{f:<45} {os.path.getsize(os.path.join(OUTPUT_DIR,f))/1024:>7.1f} KB')
for f in os.listdir(CHECKPOINT_DIR):
    print(f'  checkpoints/{f:<42} {os.path.getsize(os.path.join(CHECKPOINT_DIR,f))/1024/1024:>7.1f} MB')
print('\nDownloading...')
files.download(zip_path)
print('  Download complete!')
# ── Leaf Species Predictor ────────────────────────────────────
# Upload any leaf image and get the predicted species

from google.colab import files
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ── Load best model ───────────────────────────────────────────
print('Loading trained model...')

# Rebuild model classes (needed if kernel restarted)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base          = models.resnet18(weights=None)
        base.conv1    = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.feat_dim = base.fc.in_features
        base.fc       = nn.Identity()
        self.backbone = base
    def forward(self, x): return self.backbone(x)

class LeafClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        for p in encoder.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(encoder.feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.head(self.encoder(x))

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
FT_CKPT  = '/content/checkpoints/finetuned.pth'

ckpt        = torch.load(FT_CKPT, map_location=DEVICE)
CLASS_NAMES = ckpt['class_names']
NUM_CLASSES = len(CLASS_NAMES)

enc      = Encoder()
model    = LeafClassifier(enc, NUM_CLASSES).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()

print(f'  Model loaded')
print(f'  Device     : {DEVICE}')
print(f'  Species    : {CLASS_NAMES}')

# ── Transform for input image ─────────────────────────────────
transform = T.Compose([
    T.Grayscale(1),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

# ── Prediction function ───────────────────────────────────────
def predict(image_path):
    img        = Image.open(image_path).convert('RGB')
    tensor     = transform(img).unsqueeze(0).to(DEVICE)  # add batch dim

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]          # probabilities

    top5_probs, top5_idx = torch.topk(probs, min(5, NUM_CLASSES))

    predicted_species = CLASS_NAMES[top5_idx[0].item()]
    confidence        = top5_probs[0].item() * 100

    # ── Display image + result ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: input image
    axes[0].imshow(img)
    axes[0].set_title('Input Image', fontsize=12)
    axes[0].axis('off')

    # Right: top-5 bar chart
    species_labels = [CLASS_NAMES[i.item()] for i in top5_idx]
    prob_values    = [p.item()*100 for p in top5_probs]
    colors         = ['#4C9BE8' if i==0 else '#B5D4F4' for i in range(len(species_labels))]

    bars = axes[1].barh(species_labels[::-1], prob_values[::-1],
                        color=colors[::-1], edgecolor='white')
    for bar, val in zip(bars, prob_values[::-1]):
        axes[1].text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                     f'{val:.1f}%', va='center', fontsize=10)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Top Predictions', fontsize=12)
    axes[1].set_xlim(0, 115)

    plt.suptitle(
        f'Predicted: {predicted_species}  ({confidence:.1f}% confident)',
        fontsize=14, fontweight='bold', color='#1F4E79'
    )
    plt.tight_layout()
    plt.show()

    # ── Print result ──────────────────────────────────────────
    print('='*45)
    print(f'  Predicted Species : {predicted_species}')
    print(f'  Confidence        : {confidence:.2f}%')
    print()
    print('  Top 5 Predictions:')
    print(f'  {"Rank":<6} {"Species":<25} {"Confidence"}')
    print(f'  {"-"*45}')
    for rank, (idx, prob) in enumerate(zip(top5_idx, top5_probs), 1):
        marker = '  <-- predicted' if rank == 1 else ''
        print(f'  {rank:<6} {CLASS_NAMES[idx.item()]:<25} {prob.item()*100:.2f}%{marker}')
    print('='*45)

    return predicted_species, confidence

# ── Upload and predict ────────────────────────────────────────
print('\n' + '='*45)
print('  Upload your leaf image below')
print('='*45)
print('  Supported formats: JPG, JPEG, PNG')
print('  Click "Choose Files" button...\n')

uploaded = files.upload()

for filename in uploaded.keys():
    print(f'\nPredicting: {filename}')
    predicted, confidence = predict(filename)
