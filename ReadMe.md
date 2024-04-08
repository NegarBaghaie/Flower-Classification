# Flower Classification

We've gathered a dataset of 102 types of flowers commonly found in the United Kingdom. Each flower category contains between 40 and 258 images, totaling 8189 samples. The distribution of samples for each category is outlined in Table 1.

Dataset can be downloaded from this [website](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)


| label | label name                | count |
| ----- | ------------------------- | ----- |
| 0     | pink primrose             | 40    |
| 1     | hard-leaved pocket orchid | 60    |
| 2     | canterbury bells          | 40    |
| 3     | sweet pea                 | 56    |
| 4     | english marigold          | 65    |
| 5     | tiger lily                | 45    |
| 6     | moon orchid               | 40    |
| 7     | bird of paradise          | 85    |
| 8     | monkshood                 | 46    |
| 9     | globe thistle             | 45    |
| 10    | snapdragon                | 87    |
| 11    | colt's foot               | 87    |
| 12    | king protea               | 49    |
| 13    | spear thistle             | 48    |
| 14    | yellow iris               | 49    |
| 15    | globe-flower              | 41    |
| 16    | purple coneflower         | 85    |
| 17    | peruvian lily             | 82    |
| 18    | balloon flower            | 49    |
| 19    | giant white arum lily     | 56    |
| 20    | fire lily                 | 40    |
| 21    | pincushion flower         | 59    |
| 22    | fritillary                | 91    |
| 23    | red ginger                | 42    |
| 24    | grape hyacinth            | 41    |
| 25    | corn poppy                | 41    |
| 26    | prince of wales feathers  | 40    |
| 27    | stemless gentian          | 66    |
| 28    | artichoke                 | 78    |
| 29    | sweet william             | 85    |
| 30    | carnation                 | 52    |
| 31    | garden phlox              | 45    |
| 32    | love in the mist          | 46    |
| 33    | mexican aster             | 40    |
| 34    | alpine sea holly          | 43    |
| 35    | ruby-lipped cattleya      | 75    |
| 36    | cape flower               | 108   |
| 37    | great masterwort          | 56    |
| 38    | siam tulip                | 41    |
| 39    | lenten rose               | 67    |
| 40    | barbeton daisy            | 127   |
| 41    | daffodil                  | 59    |
| 42    | sword lily                | 130   |
| 43    | poinsettia                | 93    |
| 44    | bolero deep blue          | 40    |
| 45    | wallflower                | 196   |
| 46    | marigold                  | 67    |
| 47    | buttercup                 | 71    |
| 48    | oxeye daisy               | 49    |
| 49    | common dandelion          | 92    |
| 50    | petunia                   | 258   |
| 51    | wild pansy                | 85    |
| 52    | primula                   | 93    |
| 53    | sunflower                 | 61    |
| 54    | pelargonium               | 71    |
| 55    | bishop of llandaff        | 109   |
| 56    | gaura                     | 67    |
| 57    | geranium                  | 114   |
| 58    | orange dahlia             | 67    |
| 59    | pink-yellow dahlia?       | 109   |
| 60    | cautleya spicata          | 50    |
| 61    | japanese anemone          | 55    |
| 62    | black-eyed susan          | 54    |
| 63    | silverbush                | 52    |
| 64    | californian poppy         | 102   |
| 65    | osteospermum              | 61    |
| 66    | spring crocus             | 42    |
| 67    | bearded iris              | 54    |
| 68    | windflower                | 54    |
| 69    | tree poppy                | 62    |
| 70    | gazania                   | 78    |
| 71    | azalea                    | 96    |
| 72    | water lily                | 194   |
| 73    | rose                      | 171   |
| 74    | thorn apple               | 120   |
| 75    | morning glory             | 107   |
| 76    | passion flower            | 251   |
| 77    | lotus                     | 137   |
| 78    | toad lily                 | 41    |
| 79    | anthurium                 | 105   |
| 80    | frangipani                | 166   |
| 81    | clematis                  | 112   |
| 82    | hibiscus                  | 131   |
| 83    | columbine                 | 86    |
| 84    | desert-rose               | 63    |
| 85    | tree mallow               | 58    |
| 86    | magnolia                  | 63    |
| 87    | cyclamen                  | 154   |
| 88    | watercress                | 184   |
| 89    | canna lily                | 82    |
| 90    | hippeastrum               | 76    |
| 91    | bee balm                  | 66    |
| 92    | ball moss                 | 46    |
| 93    | foxglove                  | 162   |
| 94    | bougainvillea             | 128   |
| 95    | camellia                  | 91    |
| 96    | mallow                    | 66    |
| 97    | mexican petunia           | 82    |
| 98    | bromelia                  | 63    |
| 99    | blanket flower            | 49    |
| 100   | trumpet creeper           | 58    |
| 101   | blackberry lily           | 48    |

*Table1: This table shows the number of examples for each class*

## Preprocessing:

Since EfficientNetV2B0 serves as the base network, we've incorporated its corresponding preprocessing input function. This function is tailored to preprocess images optimally for EfficientNetV2B0.

Moreover, to ensure compatibility with pretrained models and maintain image quality, we resize images to (224, 224, 3) dimensions

Given our small and imbalanced dataset, augmentation is crucial. Augmentation parameters include:

- Random contrast adjustment (0.5 to 1.2).
- Random horizontal flipping.
- Random brightness adjustment (max_delta = 0.3).
- Random saturation adjustment (0.6 to 1.2).
- Random translation adjustment (0.12, 0.12)
- Random Rotation adjustment (0.1)
-  Image values are clipped to the range [0, 255].



## Address the problem of imbalanced data
To tackle the issue of imbalanced data, we've implemented class weights, which are applied to the loss function. These weights are computed as follows:

$$ w_i = \frac{C}{n \times c_i} $$

Where:
- $w_i$ represents the weight for class $i$,
- $C$ denotes the total number of samples,
- $c_i$ signifies the number of samples for class $i$,
- $n$ represents the total number of classes.


## Model Architecture

| Layer (type)                                      | Output Shape       | Param # |
| ------------------------------------------------- | ------------------ | ------- |
| efficientnetv2-b0 (Functio nal)                   | (None, 7, 7, 1280) | 5919312 |
| global_average_pooling2d (GlobalAveragePooling2D) | (None, 1280)       | 0       |
| dropout (Dropout)                                 | (None, 1280)       | 0       |
| dense (Dense)                                     | (None, 102)        | 130662  |

**Total params: 6049974 (23.08 MB)**\
**Trainable params: 4617362 (17.61 MB)**\
**Non-trainable params: 1432612 (5.46 MB)**

## Results

| class | precision | recall | f1-score | support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 1.00      | 1.00   | 1.00     | 8       |
| 1     | 1.00      | 1.00   | 1.00     | 12      |
| 2     | 1.00      | 1.00   | 1.00     | 8       |
| 3     | 0.92      | 1.00   | 0.96     | 11      |
| 4     | 1.00      | 1.00   | 1.00     | 13      |
| 5     | 0.90      | 1.00   | 0.95     | 9       |
| 6     | 1.00      | 1.00   | 1.00     | 8       |
| 7     | 1.00      | 1.00   | 1.00     | 17      |
| 8     | 1.00      | 0.89   | 0.94     | 9       |
| 9     | 1.00      | 1.00   | 1.00     | 9       |
| 10    | 1.00      | 1.00   | 1.00     | 17      |
| 11    | 1.00      | 0.94   | 0.97     | 17      |
| 12    | 1.00      | 1.00   | 1.00     | 9       |
| 13    | 1.00      | 1.00   | 1.00     | 9       |
| 14    | 1.00      | 1.00   | 1.00     | 9       |
| 15    | 1.00      | 1.00   | 1.00     | 8       |
| 16    | 1.00      | 1.00   | 1.00     | 17      |
| 17    | 1.00      | 1.00   | 1.00     | 16      |
| 18    | 1.00      | 1.00   | 1.00     | 9       |
| 19    | 1.00      | 1.00   | 1.00     | 11      |
| 20    | 1.00      | 1.00   | 1.00     | 8       |
| 21    | 1.00      | 1.00   | 1.00     | 11      |
| 22    | 1.00      | 1.00   | 1.00     | 18      |
| 23    | 1.00      | 1.00   | 1.00     | 8       |
| 24    | 1.00      | 1.00   | 1.00     | 8       |
| 25    | 1.00      | 1.00   | 1.00     | 8       |
| 26    | 1.00      | 1.00   | 1.00     | 8       |
| 27    | 1.00      | 1.00   | 1.00     | 13      |
| 28    | 1.00      | 1.00   | 1.00     | 15      |
| 29    | 1.00      | 0.88   | 0.94     | 17      |
| 30    | 1.00      | 1.00   | 1.00     | 10      |
| 31    | 0.78      | 0.78   | 0.78     | 9       |
| 32    | 1.00      | 1.00   | 1.00     | 9       |
| 33    | 1.00      | 1.00   | 1.00     | 8       |
| 34    | 1.00      | 1.00   | 1.00     | 8       |
| 35    | 1.00      | 1.00   | 1.00     | 15      |
| 36    | 1.00      | 1.00   | 1.00     | 21      |
| 37    | 1.00      | 1.00   | 1.00     | 11      |
| 38    | 1.00      | 0.88   | 0.93     | 8       |
| 39    | 0.93      | 1.00   | 0.96     | 13      |
| 40    | 1.00      | 1.00   | 1.00     | 25      |
| 41    | 1.00      | 1.00   | 1.00     | 11      |
| 42    | 1.00      | 0.96   | 0.98     | 26      |
| 43    | 1.00      | 1.00   | 1.00     | 18      |
| 44    | 1.00      | 1.00   | 1.00     | 8       |
| 45    | 1.00      | 1.00   | 1.00     | 39      |
| 46    | 1.00      | 1.00   | 1.00     | 13      |
| 47    | 1.00      | 1.00   | 1.00     | 14      |
| 48    | 1.00      | 1.00   | 1.00     | 9       |
| 49    | 0.95      | 1.00   | 0.97     | 18      |
| 50    | 0.96      | 1.00   | 0.98     | 51      |
| 51    | 1.00      | 1.00   | 1.00     | 17      |
| 52    | 1.00      | 1.00   | 1.00     | 18      |
| 53    | 1.00      | 1.00   | 1.00     | 12      |
| 54    | 1.00      | 1.00   | 1.00     | 14      |
| 55    | 1.00      | 1.00   | 1.00     | 21      |
| 56    | 1.00      | 1.00   | 1.00     | 13      |
| 57    | 1.00      | 1.00   | 1.00     | 22      |
| 58    | 1.00      | 1.00   | 1.00     | 13      |
| 59    | 1.00      | 1.00   | 1.00     | 21      |
| 60    | 1.00      | 1.00   | 1.00     | 10      |
| 61    | 1.00      | 1.00   | 1.00     | 11      |
| 62    | 1.00      | 1.00   | 1.00     | 10      |
| 63    | 1.00      | 1.00   | 1.00     | 10      |
| 64    | 1.00      | 1.00   | 1.00     | 20      |
| 65    | 1.00      | 1.00   | 1.00     | 12      |
| 66    | 1.00      | 1.00   | 1.00     | 8       |
| 67    | 1.00      | 1.00   | 1.00     | 10      |
| 68    | 1.00      | 1.00   | 1.00     | 10      |
| 69    | 1.00      | 1.00   | 1.00     | 12      |
| 70    | 1.00      | 1.00   | 1.00     | 15      |
| 71    | 1.00      | 1.00   | 1.00     | 19      |
| 72    | 1.00      | 0.97   | 0.99     | 38      |
| 73    | 1.00      | 1.00   | 1.00     | 34      |
| 74    | 1.00      | 1.00   | 1.00     | 24      |
| 75    | 1.00      | 1.00   | 1.00     | 21      |
| 76    | 1.00      | 1.00   | 1.00     | 50      |
| 77    | 1.00      | 1.00   | 1.00     | 27      |
| 78    | 1.00      | 1.00   | 1.00     | 8       |
| 79    | 1.00      | 1.00   | 1.00     | 21      |
| 80    | 1.00      | 1.00   | 1.00     | 33      |
| 81    | 1.00      | 1.00   | 1.00     | 22      |
| 82    | 1.00      | 1.00   | 1.00     | 26      |
| 83    | 1.00      | 0.94   | 0.97     | 17      |
| 84    | 1.00      | 1.00   | 1.00     | 12      |
| 85    | 1.00      | 1.00   | 1.00     | 11      |
| 86    | 0.92      | 1.00   | 0.96     | 12      |
| 87    | 1.00      | 0.97   | 0.98     | 30      |
| 88    | 0.97      | 1.00   | 0.99     | 36      |
| 89    | 0.94      | 0.94   | 0.94     | 16      |
| 90    | 1.00      | 1.00   | 1.00     | 15      |
| 91    | 1.00      | 1.00   | 1.00     | 13      |
| 92    | 0.90      | 1.00   | 0.95     | 9       |
| 93    | 0.97      | 0.97   | 0.97     | 32      |
| 94    | 1.00      | 1.00   | 1.00     | 25      |
| 95    | 1.00      | 0.83   | 0.91     | 18      |
| 96    | 0.81      | 1.00   | 0.90     | 13      |
| 97    | 1.00      | 1.00   | 1.00     | 16      |
| 98    | 0.92      | 1.00   | 0.96     | 12      |
| 99    | 1.00      | 1.00   | 1.00     | 9       |
| 100   | 1.00      | 1.00   | 1.00     | 11      |
| 101   | 1.00      | 0.89   | 0.94     | 9       |


|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| accuracy     |           |        | 0.99     | 1602    |
| macro avg    | 0.99      | 0.99   | 0.99     | 1602    |
| weighted avg | 0.99      | 0.99   | 0.99     | 1602    |
