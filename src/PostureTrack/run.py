from omegaconf import DictConfig, OmegaConf
import hydra
import torch
from top_down import TopDown
from pifpaf import PifPaf

@hydra.main(version_base=None, config_path="config", config_name="cfg")
def get_config(cfg: DictConfig) -> None:
    verbose=cfg.io.verbose_level
    if verbose:
        print(OmegaConf.to_yaml(cfg))
    #loading IO cfg
    source=cfg.io.source
    gt=cfg.io.gt
    path_output_vid=cfg.io.path_output_vid
    path_output_json=cfg.io.path_output_json
    verbose=cfg.io.verbose_level
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    #device=cfg.io.device
    if cfg.arch.arch_type=="topdown":
        detector_name=cfg.detector.detector_name
        detector_size=cfg.detector.detector_size
        detector_thresh=cfg.detector.bbox_thresh
        detector_cfg={"type": detector_name , "size": detector_size,    
                    "thresh": detector_thresh}
        tracker_name= cfg.tracker.tracker_name
        tracker_type=cfg.tracker.tracker_type
        tracking_conf= cfg.tracker.tracking_conf
        path_cfg= cfg.tracker.path_cfg
        path_weights= cfg.tracker.path_weights
        tracker_cfg={"name":tracker_name, "type":tracker_type ,"conf":tracking_conf,
                    "cfg":path_cfg, "weights":path_weights}
        if cfg.arch.pose_activated:
            pose_model=cfg.keypoints.pose_model
            path_weights=cfg.keypoints.path_weights
            path_config=cfg.keypoints.path_config
            activ_3D=cfg.keypoints.pose3D_activated
            pose3D_model=cfg.keypoints.pose3D_model
            path_weights3D=cfg.keypoints.path_weights3D
            path_config3D=cfg.keypoints.path_config3D
            path_output_3d=cfg.io.path_output_3Dkeypoint
            pose_cfg={"name":pose_model, "weights": path_weights, "config":path_config, "3D": activ_3D, "3Dname": pose3D_model,
                    "weights3D":path_weights3D, "config3D":path_config3D, "path_output_3D": path_output_3d}
        else:
            pose_cfg=None
        TopDown(detector_cfg, tracker_cfg, pose_cfg, verbose, device,
                source, gt, path_output_vid, path_output_json)
    elif cfg.arch.arch_type=="bottomup":
        if cfg.arch.tracker=="pifpaf_tracker":
            checkpoint=cfg.tracker.checkpoint
            decoder=cfg.tracker.decoder
            long_edge=cfg.tracker.long_edge
            show_result=cfg.tracker.show_result
            disable_cuda=cfg.tracker.disable_cuda
            PifPaf(checkpoint, decoder, long_edge, show_result, disable_cuda,
                    source, path_output_vid, path_output_json)
        else:
            print("Tracker not implemented")


if __name__ == "__main__":
    get_config()


    