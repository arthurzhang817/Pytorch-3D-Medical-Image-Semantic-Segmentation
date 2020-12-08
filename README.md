# Pytorch-3D-Medical-Image-Semantic-Segmentation

This is the release version of my private research repository. It will be updated as my research goes.

# Why do we need AI for medical image semantic segmentation?
Radiotherapy treatment planning requires accurate contours for maximizing target coverage while minimizing the toxicities to the surrounding organs at risk (OARs). The diverse expertise and experience levels of physicians introduce large intraobserver variations in manual contouring. Interobserver and intraobserver variation of delineation results in uncertainty in treatment planning, which could compromise treatment outcome. Manual contouring by physicians in current clinical practice is time-consuming, which is incapable of supporting adaptive treatment when the patient is on the couch.

## Example

|![ezgif com-gif-maker](https://user-images.githubusercontent.com/24512849/87363829-a0cec500-c537-11ea-9c74-7c94d8ba0687.gif)|![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/24512849/87363843-a6c4a600-c537-11ea-80be-4c18407cba61.gif)|![ezgif com-gif-maker (2)](https://user-images.githubusercontent.com/24512849/87363872-bc39d000-c537-11ea-866e-6f37e3ee2615.gif)|
|:-:|:-:|:-:|
|![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/24512849/87364053-31a5a080-c538-11ea-918a-4aa45dcae14e.gif)|![ezgif com-gif-maker (4)](https://user-images.githubusercontent.com/24512849/87364058-35d1be00-c538-11ea-9ffd-d2f9dcc2ca7c.gif)|![ezgif com-gif-maker (5)](https://user-images.githubusercontent.com/24512849/87364085-47b36100-c538-11ea-92ca-983231dbe1a3.gif)|
|CT Slice|Ground Truth|Prediction|

# Update Log

7/11/2020 Update

- Basic training/validation function
- Model: Deeper 3D Residual U-net

7/13/2020 Update

- Model: 3D Residual U-net
- Normalization control in dataloader

# Consider citing our paper:
Zhang, Z., Zhao, T., Gay, H., Zhang, W., & Sun, B. (2020). ARPM‐net: A novel CNN‐based adversarial method with Markov Random Field enhancement for prostate and organs at risk segmentation in pelvic CT images. Medical Physics.
