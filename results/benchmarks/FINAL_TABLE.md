# CNN benchmark -- final results (20-epoch run)

Generated from `summary_20260208_214002.csv`. Do not hand-edit; regenerate with `results/make_benchmark_artifacts.py`.

## Accuracy

| Model | MRI_tumor_binary | MRI_tumor_multiclass | MRI_ms | CT_stroke_binary |
|-------|---|---|---|---|
| resnet50 | 99.6% | 95.6% | 53.6% | 94.7% |
| resnet101 | 99.8% | 98.3% | 59.7% | 97.0% |
| vgg16 | 100.0% | 97.2% | 58.5% | 95.8% |
| densenet121 | 99.6% | 98.0% | 46.6% | 97.2% |
| densenet169 | 99.8% | 99.0% | 50.3% | 97.7% |
| mobilenet_v2 | 99.3% | 97.1% | 52.6% | 97.6% |
| EffNet_b0 | 99.8% | 98.1% | 50.7% | 97.0% |
| EffNet_b4 | 99.8% | 98.0% | 54.1% | 96.9% |


## Weighted F1

F1 with `average='weighted'`.

| Model | MRI_tumor_binary | MRI_tumor_multiclass | MRI_ms | CT_stroke_binary |
|-------|---|---|---|---|
| resnet50 | 99.6% | 95.5% | 54.4% | 94.7% |
| resnet101 | 99.8% | 98.3% | 60.3% | 97.0% |
| vgg16 | 100.0% | 97.2% | 59.8% | 95.8% |
| densenet121 | 99.6% | 98.0% | 46.5% | 97.2% |
| densenet169 | 99.8% | 99.0% | 50.9% | 97.7% |
| mobilenet_v2 | 99.3% | 97.0% | 53.5% | 97.6% |
| EffNet_b0 | 99.8% | 98.1% | 52.1% | 97.0% |
| EffNet_b4 | 99.8% | 98.0% | 53.9% | 96.9% |


## AUC

ROC-AUC: binary on the two-class tasks, one-vs-rest (`multi_class='ovr'`) on `MRI_tumor_multiclass`.

| Model | MRI_tumor_binary | MRI_tumor_multiclass | MRI_ms | CT_stroke_binary |
|-------|---|---|---|---|
| resnet50 | 0.9998 | 0.9975 | 0.6140 | 0.9876 |
| resnet101 | 1.0000 | 0.9998 | 0.6514 | 0.9934 |
| vgg16 | 1.0000 | 0.9992 | 0.6644 | 0.9850 |
| densenet121 | 0.9999 | 0.9995 | 0.5495 | 0.9949 |
| densenet169 | 1.0000 | 0.9999 | 0.5728 | 0.9946 |
| mobilenet_v2 | 1.0000 | 0.9997 | 0.5887 | 0.9951 |
| EffNet_b0 | 1.0000 | 0.9997 | 0.5652 | 0.9949 |
| EffNet_b4 | 0.9999 | 0.9996 | 0.6075 | 0.9944 |


## Average accuracy across the four tasks

| Model | Mean accuracy |
|-------|---|

| resnet101 | 88.7% |
| vgg16 | 87.9% |
| EffNet_b4 | 87.2% |
| densenet169 | 86.7% |
| mobilenet_v2 | 86.6% |
| EffNet_b0 | 86.4% |
| resnet50 | 85.9% |
| densenet121 | 85.3% |
