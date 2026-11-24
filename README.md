# Pet face detection: dataset + labeler

Goal: one clean YOLO-pose dataset (cat_face / dog_face + left eye, right eye, nose)
from pet face detection dataset-1 and the auto-labeled PetFace images.

    pip install ultralytics fastapi uvicorn python-multipart pillow imagehash
    D=/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941

## 1. First model on dataset-1
    python train_face_model.py --src-root "$D/pet_face_detection_dataset_1" --work "$D/yolo_pfd1" --prepare-only
    python check_yolo_dataset.py "$D/yolo_pfd1" --hash
    python train_face_model.py --src-root "$D/pet_face_detection_dataset_1" --work "$D/yolo_pfd1" --device 0
    M=$D/yolo_pfd1/runs/pet_faces/weights/best.pt

## 2. Auto-label PetFace (face box + 3 landmarks)
    python autolabel_folders.py run --root "$D/cat" --species cat --model "$M" --out "$D/petface_labels/cat.csv" --device 0 --half
    python autolabel_folders.py run --root "$D/dog" --species dog --model "$M" --out "$D/petface_labels/dog.csv" --device 0 --half
  check / correct the uncertain ones:
    python autolabel_folders.py review --csv "$D/petface_labels/cat.csv" --root "$D/cat" --review-dir "$D/petface_review/cat"
    python app.py --images "$D/petface_review/cat/images" --labels "$D/petface_review/cat/labels" --model "$M"
    python autolabel_folders.py merge --csv "$D/petface_labels/cat.csv" --review-dir "$D/petface_review/cat"

## 3. Merge everything into the master dataset
    python build_pet_face_dataset.py add --source pfd1 --src-root "$D/pet_face_detection_dataset_1" --out "$D/pet_faces"
    python build_pet_face_dataset.py add --source petface --src-root "$D/cat" --labels-csv "$D/petface_labels/cat.csv" --out "$D/pet_faces" --per-identity 1 --max-images 8000
    python build_pet_face_dataset.py add --source petface --src-root "$D/dog" --labels-csv "$D/petface_labels/dog.csv" --out "$D/pet_faces" --per-identity 1 --max-images 8000
    python build_pet_face_dataset.py stats --out "$D/pet_faces"

## 4. Export the clean training set, check it, train
    python export_yolo.py --master "$D/pet_faces" --out "$D/yolo_merged" --clean
    python check_yolo_dataset.py "$D/yolo_merged" --hash
    python train_face_model.py --data "$D/yolo_merged/data.yaml" --work "$D/yolo_merged" --device 0

## Files
    train_face_model.py       prepare dataset-1 / train a YOLO-pose model
    check_yolo_dataset.py     check labels + picture grids
    autolabel_folders.py      batch auto-label big folders, review, merge corrections
    app.py + static/          labeler website
    build_pet_face_dataset.py master dataset (all sources, all label info)
    export_yolo.py            master -> clean YOLO-pose train/val/test (split per animal)
    survey_folders.py         look at unknown image folders

## 4. Check label quality
    # build the reference template once from dataset-1 train
    python quality_check.py build-template --format pfd1 \
        --path "$D/pet_face_detection_dataset_1" --subsets train --out "$D/quality/template"
    # check other labels against it -> report.html lists ALL suspicious images in 4 groups
    # (wrong labels, unusual shape, borderline, image/data problems) + suspicious_images.csv
    # tip: also check --subsets train itself to clean the training data
    python quality_check.py check --format pfd1 --path "$D/pet_face_detection_dataset_1" \
        --subsets val test --template "$D/quality/template/template.json" --out "$D/quality/val_test" --selftest
    # your own labels from the website -> also fills the "Needs check" filter there
    python quality_check.py check --format labeler --path "$D/new_labels" --images "$D/new_images" \
        --template "$D/quality/template/template.json" --out "$D/quality/new_labels"

## 5. Clean the cat/ and dog/ folders (species known from the folder)
    python quality_check.py check --format csv --path "$D/petface_labels/cat.csv" --images "$D/cat" \
        --template "$D/quality/template/template.json" --out "$D/quality/cat"
    python autolabel_folders.py review --csv "$D/petface_labels/cat.csv" --root "$D/cat" \
        --review-dir "$D/petface_review/cat_quality" --files "$D/quality/cat/suspicious_files.txt"
    python app.py --images "$D/petface_review/cat_quality/images" --labels "$D/petface_review/cat_quality/labels" --model "$M"
    python autolabel_folders.py merge --csv "$D/petface_labels/cat.csv" --review-dir "$D/petface_review/cat_quality"
  (same for dog)

## 6. Face crops (cat/ and dog/ folders)
  website on a crop folder:
    python app.py --images "$D/cat" --labels "$D/new_labels" --model "$M" --species cat --single-face --pad 0.5
  compare labels with the template on the keypoints only (rebuild the template once first):
    python quality_check.py check --format labeler --path "$D/new_labels" --images "$D/cat" \
        --template "$D/quality/template/template.json" --out "$D/quality/website_cat" --crops --species cat --top 200

## 7. Same animal under several IDs (for recognition)
    python find_duplicate_ids.py --root "$D/cat" --out "$D/quality/cat_ids"
  -> id_groups.csv (groups of IDs sharing photos, merge / review), id_merge_map.json

## 8. Save reviewed images as a training set (e.g. the reviewed cats)
    python save_reviewed.py --labels "$D/new_labels" --images "$D/cat" --out "$D/reviewed_sets/cat" --species cat
  rerun any time: adds new reviews, updates changed labels, keeps the rest (--sync also drops un-reviewed ones)
  -> images/ (copies) + labels/ (YOLO-pose txt) + json/ (full labels) + manifest.csv + data.yaml + info.json

## 9. Label dogs with live shape check + statistics
    python app.py --images "$D/dog" --labels "$D/dog_labels" --model "$M" --species dog --single-face --pad 0.5 \
                  --template "$D/quality/template/template.json"
  right panel: green = normal dog keypoint layout, orange = unusual (with reasons);
  dashed circles = where the nose usually sits given the eyes (key G hides them);
  Statistics = your reviewed labels vs the template (refreshes every 10 reviews)
  later, a template from your own reviewed dogs:
    python save_reviewed.py --labels "$D/dog_labels" --images "$D/dog" --out "$D/reviewed_sets/dog" --species dog
    python quality_check.py build-template --format labeler --path "$D/reviewed_sets/dog/json" \
        --images "$D/reviewed_sets/dog/images" --out "$D/quality/template_dog_reviewed"

## 10. Fine-tune the labeler on your reviewed cats + dogs
    python finetune_labeler.py --sets "$D/reviewed_sets/cat" "$D/reviewed_sets/dog" \
        --base-model "$M" --replay "$D/yolo_pfd1" --work "$D/finetune_v2" --device 0
  - leaves out reviewed images without a face; split by animal (kept in split.csv)
  - same gray border as the website (--pad 0.5); 3000 dataset-1 images mixed in (--replay-n)
  - ends with old vs new on the same held-out images + a verdict -> <work>/eval.md
  - new model: <work>/runs/finetune/weights/best.pt  (use it only if the verdict says so)

## 11. Next batch to review (highest / lowest estimated error + random)
    python next_batch.py --labels "$D/dog_labels" --images "$D/dog" --species dog \
        --model "$D/finetune_v2/runs/finetune/weights/best.pt" \
        --old-model "$D/yolo_pfd1/runs/pet_faces/weights/best.pt" \
        --template "$D/quality/template/template.json" \
        --split "$D/finetune_v2/split.csv" --set-name dog --n 500 --device 0
  - estimates each auto-label's error (mirror test, rotation test, old vs new model, layout check)
  - batch = 60% high error, 20% low error (quick confirms), 20% random (true quality)
  - refreshes the batch drafts with the new model (never reviewed labels or drafts you edited)
  - website filter "Next batch"; prints the estimated share of auto-labels that need a correction
  "Auto-label all" in the website now asks whether to also replace unreviewed drafts
  (drafts you edited by hand are always kept).

## 12. Same process for cats + balanced evaluation
    python next_batch.py --labels "$D/new_labels" --images "$D/cat" --species cat --model "$M" \
        --old-model "$D/yolo_pfd1/runs/pet_faces/weights/best.pt" --template "$D/quality/template/template.json" \
        --split "$D/finetune_v2/split.csv" --set-name cat --n 500 --device 0
  finetune_labeler.py evaluation: every test set has the SAME number of cats and dogs
  (--no-balance to switch off), tables show all / cats / dogs (+ better, - worse), a
  "Where the new model struggles" summary names the harder animal and its hardest point,
  and the verdict refuses a model that got clearly worse on cats or on dogs.

## 13. Best keypoint threshold for the website (--kp-conf)
    python tune_kp_conf.py --work "$D/finetune_v3" --model "$D/finetune_v3/runs/finetune/weights/best.pt" --device 0
  tries --kp-conf 0.30 ... 0.95 on your validation faces (same cats/dogs as the evaluation) and
  counts points you would have to move / add / delete; then use the best value in
  app.py --kp-conf X  and  next_batch.py --kp-conf X

## 14. Save only the trustworthy labels of the whole cat/ and dog/ folders
    M=$D/finetune_v3/runs/finetune/weights/best.pt
    python save_good_labels.py --root "$D/cat" --species cat --model "$M" --reviewed "$D/reviewed_sets/cat" \
        --template "$D/quality/template/template.json" --out "$D/final_labels/cat" --kp-conf 0.8 --device 0 --half
    (same for dog; test first with --limit 2000; resumable)
  -> good.csv (same columns as autolabel_folders.py + est_error), left_out.csv (reason), summary.json
  Note: autolabel_folders.py fed RGB arrays where ultralytics expects BGR until 2026-11-24 (fixed);
  CSVs it made before that were labeled with red/blue swapped.
