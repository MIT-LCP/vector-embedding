import torch
from torchvision.models.video import r3d_18, R3D_18_Weights
from tqdm import tqdm
import sys
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
sys.path.insert(0, "vector-embedding/echo/echo_r3d_transformer")
import utils_r3d

"""
to use this script, the ECHO_PRIME_ENCODER_PATH, DICOMPATH_CSV and VE_OUTPUT_PATH variables below.
"""
R3D_ENCODER_PATH = "**/r3d_binary_111723.pt"
DICOMPATH_CSV = "**/**.csv" # file containing dicom_path column as "dicom_path" with view classifications as "view"
VE_OUTPUT_PATH = "**/ve_r3d_transformer.csv"



def main(filepaths, model_path, output_path):
    weights1 = R3D_18_Weights.DEFAULT
    model1 = r3d_18(weights=weights1)
    model1.eval()
    model_r3d = utils_r3d.r3dmodel(model1)
    model = model_r3d
    model_parameters = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_parameters)
    # modify it with identity layers
    model._modules['new_layer1'] = utils_r3d.Identity()
    model._modules['new_layer2'] = utils_r3d.Identity()
    model.to(device) 

    with open(output_path, 'w') as f:
        ve_columns = [f"ve{str(i+1).zfill(3)}" for i in range(400)]
        header = ["dicom_path"] + ve_columns
        f.write(",".join(header) + "\n")

    for i, file_path in enumerate(tqdm(filepaths)):
        if file_path.endswith('.dcm'):    
            echocardiogram_tensor = utils_r3d.dicom_to_tensor(file_path)
            echocardiogram_tensor = echocardiogram_tensor.unsqueeze(2).permute(2, 1, 0, 3, 4)
            
            with torch.no_grad():
                echocardiogram_tensor = echocardiogram_tensor.to(device)
                embeddings = model(echocardiogram_tensor).cpu().numpy()
            
            with open(output_path, 'a') as f:
                embedding_str = ",".join([f"{x:.16e}" for x in embeddings.flatten()])
                row_data = f"{file_path},{embedding_str}\n"
                f.write(row_data)

if __name__ == '__main__':
    echo_direction = pd.read_csv(DICOMPATH_CSV)
    dicom_a4c = echo_direction[echo_direction.view == 'A4C'].reset_index(drop=True)
    dicom_a4c_paths = dicom_a4c['dicom_path'].tolist()  
    
    main(dicom_a4c_paths, 
          R3D_ENCODER_PATH,
          VE_OUTPUT_PATH)