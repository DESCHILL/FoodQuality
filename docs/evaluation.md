# Evaluation record

## Reported tomato image

The supplied image is a red tomato with a clean surface. The original FoodClass model returned spoiled with a score of 79.6%; the source-trained merged model returned unknown. SIC EfficientNet identified tomato and favoured normal appearance. This image was not added to training and no filename/image-specific rule was introduced.

The three-source pipeline identifies tomato. Adjusted recognition scores on this image:

| Source | Food | Score |
|---|---|---:|
| FoodQuality | Tomato | 60.0% |
| SIC EfficientNet | Tomato | 82.3% |
| SIC YOLO | Tomato | 36.1% |

Appearance remains **unknown** because evidence is insufficient/conflicting. This removes the confident spoiled claim but does not constitute a successful fresh classification. The original unadjusted YOLO score is 32.1%, below its calibration sample cutoff; the adjusted number is an extrapolated ranking score.

## Internal evaluation, 990 images

All three models ran on every image. The final deterministic quality policy was replayed over those exact stored outputs after correcting defect/spoiled compatibility; model outputs were not regenerated or fitted on the test labels.

| Metric | Result |
|---|---:|
| Food name correct | 987 / 990 = 99.70% |
| State conclusion emitted | 943 / 990 = 95.25% |
| Food + state correct among accepted | 930 / 943 = 98.62% |
| Correct and accepted over all images | 930 / 990 = 93.94% |
| Unknown | 47 |
| Incorrect accepted conclusions | 13 |

These are internal development results on public-source labels. The set has been reused during development; it is not an untouched external benchmark. The overlap with SIC's original training data is unknown. There is no verified multi-object bounding-box benchmark or guarantee for unseen camera/background conditions. Mutton still has very few independent groups.

## Alternatives tested

Cosine-normalized logistic heads were tested at C = 0.1, 1, 10, 100, and cosine k-NN at k = 3, 9, 21. The best logistic variant reached validation macro-F1 96.76%, below the deployed classifier's 98.13% on the same original validation split, so it was not substituted on the strength of its test score.

AgriFreshNET supplied 631 additional non-near-duplicate tomato images: 457 fresh and 174 rotten. The expanded experiment contains 6,823 images, split 4,604 train / 1,098 validation / 1,121 test. Source labels remain unreviewed; capture date is a grouping surrogate, not proof of distinct specimens. Semi-fresh and unparseable acquisition dates are excluded.

On the expanded tomato test subset (201 images), the candidate reached 92.04% joint accuracy versus 68.16% for the deployed classifier. However, validation spoiled recall declined from 96.71% to 95.72% overall. The candidate did not pass the release guard and was retained as an experiment. A higher-spoiled-weight run also failed the validation recall requirement and produced no deployable registry.

## Verification

- 61 unit/integration tests passed locally.
- The real uploaded tomato was checked through the Streamlit automatic-upload path.
- The web result contains all three source scores and does not duplicate history on rerun.
- GitHub Actions runs the same source tests on Windows with CPU dependencies; see the repository Actions page for each commit's result.

Weights, raw images and per-image reports remain local. The repository contains source, source manifests, experiment scripts and this aggregate report; it does not redistribute third-party model weights or claim that cloning code alone supplies the datasets.
