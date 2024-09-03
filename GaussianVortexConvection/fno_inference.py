import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import hydra
import os
import vtk
import joblib
from vtk.util.numpy_support import numpy_to_vtk
from modulus.models.fno import FNO  # Adjust import based on your model location
from hydra.utils import to_absolute_path
from modulus.launch.utils import load_checkpoint
from sklearn.preprocessing import StandardScaler

class FNOData(Dataset):
    def __init__(self, x):
        self.x = x
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i]

def load_model(checkpoint_path, cfg, device):
    model = FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=cfg.arch.fno.fno_modes,
        padding=cfg.arch.fno.padding,
    ).to(device)
    
    # Load checkpoint using the modulus utility function
    load_checkpoint(path=checkpoint_path, device=device, models=model)
    model.eval()
    return model

def perform_inference(model, data_loader, device, scalars):
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch.to(device)
            pred_batch = model(x_batch)

            # Separate the channels
            pred_batch_u = pred_batch[:,0:1,:,:]
            pred_batch_v = pred_batch[:,1:,:,:]

            # Inverse transform the predictions for each channel
            pred_batch_u_np = pred_batch_u.cpu().numpy().reshape(-1, 1)
            pred_original_u = scalars['u'].inverse_transform(pred_batch_u_np)
            pred_original_u = pred_original_u.reshape(pred_batch_u.shape)

            pred_batch_v_np = pred_batch_v.cpu().numpy().reshape(-1, 1)
            pred_original_v = scalars['v'].inverse_transform(pred_batch_v_np)
            pred_original_v = pred_original_v.reshape(pred_batch_v.shape)

            # Combine the channels back together
            pred_original = np.concatenate((pred_original_u, pred_original_v), axis=1)

            predictions.append(pred_original)

    predictions = np.concatenate(predictions, axis=0)
    return predictions

def save_vtk(predictions, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over timesteps
    for t in range(predictions.shape[0]):
        timestep_pred = predictions[t, 0]  # Selecting the prediction for timestep t
        dimensions = timestep_pred.shape  # Shape of the prediction
        
        # Create VTK grid and set points
        points = vtk.vtkPoints()
        values = vtk.vtkDoubleArray()
        values.SetName("PredictedField")

        for z in range(dimensions[2]):
            for y in range(dimensions[1]):
                for x in range(dimensions[0]):
                    points.InsertNextPoint(x, y, z)
                    values.InsertNextValue(timestep_pred[x, y, z])

        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(dimensions)
        grid.SetPoints(points)
        grid.GetPointData().SetScalars(values)

        # Write to VTK file
        writer = vtk.vtkStructuredGridWriter()
        vtk_filename = os.path.join(output_folder, f"predicted_field_timestep_{t}.vtk")
        writer.SetFileName(vtk_filename)
        writer.SetInputData(grid)
        writer.Write()

        print(f"Saved {vtk_filename}")

@hydra.main(version_base="1.3", config_path=".", config_name="config_fno.yaml")
def main(cfg):
    input_file_path = to_absolute_path("data.npy")  # Convert to absolute path
    scalar_path = to_absolute_path('scalars.pkl')
    checkpoint_path = to_absolute_path(cfg.scheduler.checkpoint_path)

    # Load Data
    try:
        scalars = joblib.load(scalar_path)
        data = np.load(input_file_path)
        data_u = data[:,:,0:1,:,:]
        data_v = data[:,:,1:,:,:]

        # Reshape data so that each sample is a separate row
        reshaped_data_u = data_u.reshape(-1,1)
        reshaped_data_v = data_v.reshape(-1,1)

        # Fit and transform the data
        data_u_t = scalars['u'].fit_transform(reshaped_data_u)
        data_v_t = scalars['v'].fit_transform(reshaped_data_v)

        # Reshape the data back to its original shape
        data_u = data_u_t.reshape(data_u.shape)
        data_v = data_v_t.reshape(data_v.shape)
        data = np.concatenate((data_u, data_v), axis=2)

        x = data[:,0,:,:,:]
        print("x.shape:", x.shape)
    except FileNotFoundError:
        print(f"File {input_file_path} not found.")
        return

    # Initialize Dataset and DataLoader
    inference_dataset = FNOData(torch.tensor(x).float())
    inference_loader = DataLoader(inference_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, cfg, device)

    # Perform inference
    predictions = perform_inference(model, inference_loader, device, scalars)
    print("Predictions.shape:", predictions.shape)

    # Create the sanity_check directory if it doesn't exist
    if not os.path.exists('sanity_check'):
        os.makedirs('sanity_check')

    # Loop over the first index of the predictions
    for i in range(predictions.shape[0]):
        # Select the data for the current index
        data = predictions[i, :, :, :]

        # Create a meshgrid for x and y coordinates
        x_coords = np.linspace(0, 4, data.shape[1])
        y_coords = np.linspace(0, 1, data.shape[2])
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

        # Save the data in the .dat format
        with open(f'sanity_check/data_{i}.dat', 'w') as f:
            f.write('TITLE = "Vortex Convect"\n')
            f.write('VARIABLES = "X", "Y", "u", "v"\n')
            f.write(f'ZONE T="BIG ZONE", I={data.shape[1]}, J={data.shape[2]}, DATAPACKING=POINT\n')
            for j in range(data.shape[2]):
                for k in range(data.shape[1]):
                    f.write(f"{X[k, j]} {Y[k, j]} {data[0, k, j]} {data[1, k, j]}\n")

    # Save predictions as numpy array
    np.save("predictions.npy", predictions)

if __name__ == "__main__":
    main()
