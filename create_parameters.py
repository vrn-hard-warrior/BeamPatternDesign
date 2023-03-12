# Create parameters for DeepMIMO dataset forming
import numpy as np

parameters = {'OFDM': {'RX_filter': 0,
                       'bandwidth': 0.5,
                       'subcarriers': 512,
                       'subcarriers_limit': 512,
                       'subcarriers_sampling': 1},
              'OFDM_channels': 1,
              'active_BS': np.array([3]).tolist(),
              'bs_antenna': {'radiation_pattern': 'isotropic',
                             'shape': np.array([32, 1, 1]).tolist(),
                             'spacing': 0.5},
              'dataset_folder': r'E:/Диссертация/Модели/BeamPatternDesign/data/',
              'dynamic_settings': {'first_scene': 1, 'last_scene': 1},
              'enable_BS2BS': 0,
              'num_paths': 5,
              'row_subsampling': 1,
              'scenario': 'O1_28B',
              'ue_antenna': {'radiation_pattern': 'isotropic',
                             'shape': np.array([1, 1, 1]).tolist(),
                             'spacing': 0.5},
              'user_row_first': 700,
              'user_row_last': 700,
              'user_subsampling': 1}


if __name__ == "__main__":
    import json
    
    with open(r'data/parameters.json', 'w') as outfile:
        json.dump(parameters, outfile, ensure_ascii = False, indent = 4)