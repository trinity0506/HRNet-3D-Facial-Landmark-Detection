import pandas as pd
import json
import numpy as np
import SimpleITK as sitk

def extract_nifti_parameters(nifti_path):
    nifti_img = sitk.ReadImage(nifti_path)
    affine = np.array(nifti_img.GetDirection()).reshape(3, 3)  
    origin = np.array(nifti_img.GetOrigin())  
    spacing = np.array(nifti_img.GetSpacing())  
    return affine, origin, spacing

def convert_xyz_to_ras_json(input_csv, output_json, affine, origin, spacing):

    # 读取输入 CSV 文件
    df = pd.read_csv(input_csv)

    # 检查必要列是否存在
    required_columns = ["x_voxel", "y_voxel", "z_voxel", "reason"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"输入文件缺少必要列: {col}")

    # 构建 JSON 数据结构
    json_data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "RAS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(df),
                "controlPoints": [],
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 1.0],
                    "selectedColor": [1.0, 0.5, 0.5],
                    "activeColor": [0.4, 1.0, 0.0],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "textScale": 3.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 2.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": True,
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,
                    "translationHandleVisibility": True,
                    "rotationHandleVisibility": True,
                    "scaleHandleVisibility": False,
                    "interactionHandleScale": 3.0,
                    "snapMode": "toVisibleSurface"
                }
            }
        ]
    }

    # 添加控制点信息
    control_points = []
    for index, row in df.iterrows():
        # 将 voxel 坐标转换为 RAS 坐标
        voxel_coord = np.array([row["x_voxel"], row["y_voxel"], row["z_voxel"]])
        ras_coord = origin + np.dot(affine, voxel_coord * spacing)

        control_points.append({
            "id": str(index + 1),
            "label": f"Point_{index + 1}",
            "description": row["reason"],
            "associatedNodeID": "",
            "position": ras_coord.tolist(),
            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined"
        })

    json_data["markups"][0]["controlPoints"] = control_points

    # 保存到 JSON 文件
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    print(f"JSON 文件已保存到 {output_json}")

if __name__ == "__main__":
    # 从实际的NIfTI文件中提取参数
    nifti_path = "privacydata.nii.gz"
    affine, origin, spacing = extract_nifti_parameters(nifti_path)



    affine[0, :] *= -1
    affine[1, :] *= -1
    affine[2, :] *= -1
    affine[:, [2, 1]] = affine[:, [1, 2]]
    origin[2] += 250 
    origin[0] += 250 
    origin[1] += -20 
    origin[1] *= 0.9
    # 转换CSV到JSON
    convert_xyz_to_ras_json("3D_z_face.csv", "D:/CT/CT of my skull/landmarks.json", affine, origin, spacing)