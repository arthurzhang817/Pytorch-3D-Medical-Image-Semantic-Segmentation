# Pytorch-3D-Medical-Image-Semantic-Segmentation

This is the release version of my private research repository. It will be updated as my research goes.

# Why do we need AI for medical image semantic segmentation?
Radiotherapy treatment planning requires accurate contours for maximizing target coverage while minimizing the toxicities to the surrounding organs at risk (OARs). The diverse expertise and experience levels of physicians introduce large intraobserver variations in manual contouring. Interobserver and intraobserver variation of delineation results in uncertainty in treatment planning, which could compromise treatment outcome. Manual contouring by physicians in current clinical practice is time-consuming, which is incapable of supporting adaptive treatment when the patient is on the couch.

## Example
|![patient_1_000173](https://user-images.githubusercontent.com/24512849/87240071-be871780-c3db-11ea-8c04-afb9571c18b3.png)|![patient_1_000173_target](https://user-images.githubusercontent.com/24512849/87240070-be871780-c3db-11ea-8b24-f5d6fdf29ac8.png)|![patient_1_000173_pred_ARPC_net](https://user-images.githubusercontent.com/24512849/87240072-be871780-c3db-11ea-8e00-2629bc82bc58.png)|![patient_1_000173_pred_ARPC_net_overlay_comp](https://user-images.githubusercontent.com/24512849/87240069-bdee8100-c3db-11ea-9481-ef6b5e25b545.png)|
|:-:|:-:|:-:|:--:|
|CT Slice|Target Mask|Prediction Mask|Contour Overlay (prediction in red; target in green)| 

# Update Log

7/11/2020 Update

- Basic training/validation function
- Model: 3D Residual U-net
