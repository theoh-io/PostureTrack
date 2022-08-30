import torch
from mmtrack.apis import init_model
import os


class BaseTracker():
    def __init__(self, tracker_name="", tracking_conf=None, path_cfg_tracker="", path_weights_tracker="", device='gpu', verbose="False"):
       
        desired_device = device
        path_config=os.path.abspath(os.path.join(os.getcwd(),path_cfg_tracker))
        assert os.path.exists(path_config), f"path to tracker config doesn't exists {path_config}"
        path_model=os.path.abspath(os.path.join(os.getcwd(),path_weights_tracker))
        assert os.path.exists(path_model), f"path to tracker weights doesn't exists {path_model}"


        cpu = 'cpu' == desired_device
        cuda = not cpu and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.tracker = init_model(path_config, path_model, self.device) 
        self.conf_thresh=tracking_conf
        self.frame=0
        self.verbose=verbose
        if self.verbose:
            print(f"-> Using tracker {tracker_name} from MMTracking")


    def forward(self):
        print("forward method for base Tracker is not implemented !!")
        pass

