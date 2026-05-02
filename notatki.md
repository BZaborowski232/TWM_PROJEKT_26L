
## Co wykonano:

[x] Działający pre-processing i prosta segmentacja — zrealizowano w dwóch wariantach: automatyczny pre-processing wejścia modelu YOLO oraz klasyczny moduł ROI/HSV. W module klasycznym obraz jest wczytywany, wycinany jest obszar zainteresowania obejmujący płyn w butelce, a następnie wykonywana jest konwersja do przestrzeni HSV i proste progowanie pikseli na podstawie barwy oraz jasności.

Pre-processing YOLO:
- wczytanie obrazu,
- skalowanie obrazu do rozmiaru wejściowego modelu,
- normalizacja pikseli,
- przygotowanie tensora wejściowego.

Pre-processing klasyczny ROI/HSV:
- wczytanie obrazu,
- wycięcie obszaru zainteresowania ROI,
- konwersja obrazu do przestrzeni HSV.

Prosta segmentacja:
- progowanie pikseli w ROI na podstawie jasności i koloru,
- wykrywanie pikseli ciemnych oraz pikseli o brązowym odcieniu,
- wyznaczenie cech takich jak `dark_ratio`, `brown_ratio`, `mean_saturation`, `mean_value`.


Jak wyglądało w terminalu jak uruchomiłam yolo_train.py

(.venv) PS D:\STUDIA MGR PW\TWM\TWM_Projekt> & "D:/STUDIA MGR PW/TWM/TWM_Projekt/TWM_Projekt_26L/.venv/Scripts/python.exe" "d:/STUDIA MGR PW/TWM/TWM_Projekt/TWM_PROJEKT_26L/implementation/yolo/yolo_train.py"
Ultralytics 8.4.45  Python-3.11.9 torch-2.11.0+cpu CPU (AMD Ryzen 7 5800H with Radeon Graphics)
engine\trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\implementation\yolo\data.yaml, degrees=0.0, deterministic=True, device=cpu, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=10, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8n.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=yolo_train, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\outputs, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\outputs\yolo_train, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=7

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    752677  ultralytics.nn.modules.head.Detect           [7, 16, None, [64, 128, 256]] 
Model summary: 130 layers, 3,012,213 parameters, 3,012,197 gradients, 8.2 GFLOPs

Transferred 319/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
train: Fast image access  (ping: 0.20.0 ms, read: 581.2126.2 MB/s, size: 216.7 KB)
train: Scanning D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\dataset_yolo\train\labels.cache... 960 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 960/960  0.0s
val: Fast image access  (ping: 0.20.0 ms, read: 581.531.8 MB/s, size: 217.6 KB)
val: Scanning D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\dataset_yolo\val\labels.cache... 240 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 240/240  0.0s
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.000909, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
Plotting labels to D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\outputs\yolo_train\labels.jpg... 
Image sizes 640 train, 640 val
Using 0 dataloader workers
Logging results to D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\outputs\yolo_train
Starting training for 10 epochs...
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/10         0G     0.5453      2.251      0.923         12        640: 100% ━━━━━━━━━━━━ 120/120 2.6s/it 5:07
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.1s/it 16.9s
                   all        240        367      0.869       0.66      0.768      0.678

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/10         0G     0.4423      1.195     0.8543         11        640: 100% ━━━━━━━━━━━━ 120/120 2.5s/it 4:56
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.1s/it 17.0s
                   all        240        367      0.936       0.89       0.92      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/10         0G     0.4072     0.9218     0.8389         13        640: 100% ━━━━━━━━━━━━ 120/120 2.4s/it 4:48
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.1s/it 16.5s
                   all        240        367      0.989      0.906       0.98      0.844

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/10         0G     0.4076     0.7906     0.8394         12        640: 100% ━━━━━━━━━━━━ 120/120 2.5s/it 4:55
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.1s/it 16.3s
                   all        240        367      0.981      0.978      0.991      0.894

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/10         0G     0.3694     0.6661     0.8267         10        640: 100% ━━━━━━━━━━━━ 120/120 2.4s/it 4:54
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.1s/it 16.2s
                   all        240        367      0.962      0.988      0.991      0.888

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/10         0G     0.3396     0.5744     0.8164         12        640: 100% ━━━━━━━━━━━━ 120/120 2.5s/it 4:55
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.1s/it 16.1s
                   all        240        367      0.958      0.989       0.99      0.907

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/10         0G     0.3087     0.5158     0.8013         12        640: 100% ━━━━━━━━━━━━ 120/120 2.6s/it 5:18
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.2s/it 18.0s
                   all        240        367      0.973      0.998      0.995      0.923

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/10         0G      0.285     0.4626     0.7998         10        640: 100% ━━━━━━━━━━━━ 120/120 2.7s/it 5:19
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.2s/it 18.2s
                   all        240        367       0.98      0.994      0.995      0.919

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/10         0G     0.2794     0.4364     0.7989         13        640: 100% ━━━━━━━━━━━━ 120/120 2.7s/it 5:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.2s/it 17.8s
                   all        240        367      0.983      0.997      0.995      0.935

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/10         0G     0.2583       0.41      0.792         10        640: 100% ━━━━━━━━━━━━ 120/120 2.7s/it 5:26
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.3s/it 19.6s
                   all        240        367      0.978      0.998      0.995      0.937

10 epochs completed in 0.900 hours.
Optimizer stripped from D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\outputs\yolo_train\weights\last.pt, 6.2MB
Optimizer stripped from D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\outputs\yolo_train\weights\best.pt, 6.2MB

Validating D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\outputs\yolo_train\weights\best.pt...
Ultralytics 8.4.45  Python-3.11.9 torch-2.11.0+cpu CPU (AMD Ryzen 7 5800H with Radeon Graphics)
Model summary (fused): 73 layers, 3,007,013 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 15/15 1.1s/it 16.7s
                   all        240        367      0.978      0.998      0.995      0.937
                  good        227        227      0.997          1      0.995      0.992
          wrong_bottle         13         13      0.949          1      0.995      0.995
           underfilled         40         40      0.983          1      0.995      0.958
                no_cap         29         29      0.977          1      0.995      0.871
             loose_cap         22         22      0.969          1      0.995      0.924
                debris         23         23       0.97          1      0.995       0.99
         damaged_label         13         13          1      0.986      0.995       0.83
Speed: 1.0ms preprocess, 52.5ms inference, 0.0ms loss, 0.9ms postprocess per image
Results saved to D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\outputs\yolo_train





Po zakończeniu treningu wykonano predykcję na niezależnym zbiorze testowym składającym się z 300 obrazów. Wyniki zostały zapisane w folderze `outputs/yolo_predictions`, a odpowiadające im etykiety predykcji w formacie YOLO zapisano w folderze `outputs/yolo_predictions/labels`. Predykcja potwierdziła poprawne działanie modelu na przykładowych obrazach testowych — dla obrazów z defektami model zwracał zarówno klasę `good`, oznaczającą wykrytą butelkę, jak i odpowiadającą klasę defektu, np. `damaged_label`, `debris`, `loose_cap`, `no_cap` lub `underfilled`.


W terminalu wyglądało to tak:


image 1/300 D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\dataset_yolo\test\images\damaged_label_0004_20260210_151144.jpg: 384x640 1 good, 1 damaged_label, 85.5ms
image 2/300 D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\dataset_yolo\test\images\damaged_label_0005_20260210_151147.jpg: 384x640 1 good, 1 damaged_label, 57.6ms
image 3/300 D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\dataset_yolo\test\images\damaged_label_0012_20260210_151205.jpg: 384x640 1 good, 1 damaged_label, 55.6ms
..
image 299/300 D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\dataset_yolo\test\images\wrong_bottle_0090_20260210_153329.jpg: 384x640 1 wrong_bottle, 58.9ms
image 300/300 D:\STUDIA MGR PW\TWM\TWM_Projekt\TWM_PROJEKT_26L\dataset_yolo\test\images\wrong_bottle_0091_20260210_153331.jpg: 384x640 1 wrong_bottle, 61.6ms
Speed: 2.9ms preprocess, 58.4ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)





Plik	Co oznacza w outputs:
args.yaml	zapis parametrów treningu, np. liczba epok, batch, ścieżka do danych, rozmiar obrazów
results.csv	tabela z wynikami z każdej epoki
results.png	wykres pokazujący, jak zmieniały się straty i metryki podczas treningu
labels.jpg	podgląd rozkładu etykiet w zbiorze, czyli ile jest klas i jak wyglądają bounding boxy
confusion_matrix.png	macierz pomyłek — pokazuje, które klasy model myli z którymi
confusion_matrix_normalized.png	macierz pomyłek w wersji znormalizowanej, czyli procentowo
BoxP_curve.png	wykres Precision dla detekcji ramek
BoxR_curve.png	wykres Recall dla detekcji ramek
BoxF1_curve.png	wykres F1-score
BoxPR_curve.png	wykres Precision-Recall
train_batch0.jpg, train_batch1.jpg, train_batch2.jpg	przykładowe obrazy treningowe widziane przez model podczas uczenia
val_batch0_labels.jpg	obrazy walidacyjne z prawdziwymi etykietami
val_batch0_pred.jpg	te same obrazy walidacyjne, ale z predykcjami modelu
val_batch1_labels.jpg, val_batch1_pred.jpg	kolejna porcja obrazów walidacyjnych: etykiety prawdziwe vs predykcje
val_batch2_labels.jpg, val_batch2_pred.jpg	kolejna porcja walidacyjna

