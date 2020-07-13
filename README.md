# Pytorch-3D-Medical-Image-Semantic-Segmentation

This is the release version of my private research repository. It will be updated as my research goes.

# Why do we need AI for medical image semantic segmentation?
Radiotherapy treatment planning requires accurate contours for maximizing target coverage while minimizing the toxicities to the surrounding organs at risk (OARs). The diverse expertise and experience levels of physicians introduce large intraobserver variations in manual contouring. Interobserver and intraobserver variation of delineation results in uncertainty in treatment planning, which could compromise treatment outcome. Manual contouring by physicians in current clinical practice is time-consuming, which is incapable of supporting adaptive treatment when the patient is on the couch.

## Example

|![patient_1_000146](https://user-images.githubusercontent.com/24512849/87240169-c5625a00-c3dc-11ea-88c2-9147893ef1f8.png)|![patient_1_000146_target](https://user-images.githubusercontent.com/24512849/87240172-c5faf080-c3dc-11ea-89f2-8f46fa6ca3d2.png)|![patient_1_000146_pred_ARPC_net](https://user-images.githubusercontent.com/24512849/87240170-c5625a00-c3dc-11ea-8649-7560def4271b.png)|![patient_1_000146_pred_ARPC_net_overlay_comp](https://user-images.githubusercontent.com/24512849/87240171-c5625a00-c3dc-11ea-9f79-d3590715b2d3.png)|
|:-:|:-:|:-:|:--:|
|![patient_1_000166](https://user-images.githubusercontent.com/24512849/87240173-c5faf080-c3dc-11ea-9d4d-4f77a0537355.png)|![patient_1_000166_target](https://user-images.githubusercontent.com/24512849/87240176-c6938700-c3dc-11ea-9824-342bc189c969.png)|![patient_1_000166_pred_ARPC_net](https://user-images.githubusercontent.com/24512849/87240174-c5faf080-c3dc-11ea-9ff0-574204d1d659.png)|![patient_1_000166_pred_ARPC_net_overlay_comp](https://user-images.githubusercontent.com/24512849/87240175-c6938700-c3dc-11ea-97b0-f1d9f5356b6e.png)|
|![patient_1_000173](https://user-images.githubusercontent.com/24512849/87240071-be871780-c3db-11ea-8c04-afb9571c18b3.png)|![patient_1_000173_target](https://user-images.githubusercontent.com/24512849/87240070-be871780-c3db-11ea-8b24-f5d6fdf29ac8.png)|![patient_1_000173_pred_ARPC_net](https://user-images.githubusercontent.com/24512849/87240072-be871780-c3db-11ea-8e00-2629bc82bc58.png)|![patient_1_000173_pred_ARPC_net_overlay_comp](https://user-images.githubusercontent.com/24512849/87240069-bdee8100-c3db-11ea-9481-ef6b5e25b545.png)|
|![patient_1_000190](https://user-images.githubusercontent.com/24512849/87240177-c6938700-c3dc-11ea-8be5-36d4bedb5ed5.png)|![patient_1_000190_target](https://user-images.githubusercontent.com/24512849/87240227-3ace2a80-c3dd-11ea-8036-37ae003515e5.png)|![patient_1_000190_pred_ARPC_net](https://user-images.githubusercontent.com/24512849/87240178-c6938700-c3dc-11ea-8ac4-5b3c6ca628ba.png)|![patient_1_000190_pred_ARPC_net_overlay_comp](https://user-images.githubusercontent.com/24512849/87240225-3570e000-c3dd-11ea-9cdf-b333bcb178ad.png)|
|CT Slice|Ground Truth|Prediction|Contour Overlay| 

# Update Log

7/11/2020 Update

- Basic training/validation function
- Model: Deeper 3D Residual U-net

7/13/2020 Update

- Model: 3D Residual U-net
