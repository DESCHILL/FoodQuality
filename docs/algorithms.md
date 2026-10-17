# Recognition and appearance evaluation

Three inference sources run on every submitted image: FoodQuality's source-trained classifier, SIC EfficientNet, and SIC YOLO. Recognition returns the highest adjusted score among their candidates. Recognition scores and quality scores have different meanings; a confident food name does not establish freshness.

Classifier temperatures minimize negative log likelihood on validation. The SIC food score marginalizes its appearance labels. YOLO scores use a sigmoid fit for image-level food agreement on retained validation boxes, not IoU/localization accuracy; scores below the calibration sample cutoff remain extrapolations. None of these internal adjustments guarantees calibration on arbitrary user images.

Quality evidence is evaluated separately for the selected food. Low confidence and contradictory states return unknown. SIC's generic visible-defect label is not automatically equated to spoilage. Quality claims outside the nine validated ingredients remain unsupported.

References examined:

- [Temperature scaling](https://github.com/gpleiss/temperature_scaling): algorithm reference; legacy package is unmaintained and is not installed.
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html): validation-only fitting and interpretation of scores.
- [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models): modern backbones, pretrained weights and training recipes. Adding a heavier backbone does not by itself address biased freshness labels.

The experiments include L2-normalized logistic heads (four regularization strengths), cosine nearest neighbours (three neighbourhood sizes), confidence calibration, and additional source-labelled tomatoes. Validation selects candidates; the regression photo supplied by the user is never added to training or assigned an image-specific rule.

Additional data: [AgriFreshNET V1](https://data.mendeley.com/datasets/42m5tb7yv9/1), Fouzia Risdin, Hafsha Salauddin, Tasniya aziza tufa and Md Jakir Hossain, CC BY 4.0. Fresh and Rotten tomato folders are used; Semi-Fresh is excluded. Capture-date grouping keeps the same acquisition date in one split even when it contains multiple labels. Unknown dates are not guessed. Original specimen identity remains unavailable.
