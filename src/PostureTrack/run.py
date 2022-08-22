from omegaconf import DictConfig, OmegaConf
import hydra
from top_down import TopDown
from pifpaf import PifPaf

@hydra.main(version_base=None, config_path="config", config_name="cfg")
def get_config(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    #loading IO cfg
    source=cfg.io.source
    gt=cfg.io.gt
    path_output_vid=cfg.io.path_output_vid
    path_output_json=cfg.io.path_output_json
    verbose=cfg.io.verbose
    if cfg.arch.arch_type=="topdown":
        print("top-down architecture")
        if cfg.arch.tracker_type=="SOT":
            print("using SOT Tracker")
            detector_type=cfg.detector.detector_type
            detector_size=cfg.detector.detector_size
            detector_thresh=cfg.detector.bbox_thresh
            detector_cfg={"type": detector_type , "size": detector_size,    
                        "thresh": detector_thresh}
            tracker_type= cfg.tracker.tracker_type
            tracking_conf= cfg.tracker.tracking_conf
            path_cfg= cfg.tracker.path_cfg
            path_weights= cfg.tracker.path_weights
            tracker_cfg={"type":tracker_type ,"conf":tracking_conf,
                     "cfg":path_cfg, "weights":path_weights}
            TopDown(detector_cfg, tracker_cfg, verbose,
                    source, gt, path_output_vid, path_output_json)
        elif cfg.arg.tracker_type=="MOT":
            print("using MOT Tracker")
            print("currently not supported")
            exit()
    elif cfg.arch.arch_type=="bottomup":
        print("bottom-up architecture")
        if cfg.arch.tracker=="pifpaf":
            print("using pifpaf Tracker")
            checkpoint=cfg.tracker.checkpoint
            decoder=cfg.tracker.decoder
            long_edge=cfg.tracker.long_edge
            show_result=cfg.tracker.show_result
            disable_cuda=cfg.tracker.disable_cuda
            PifPaf(checkpoint, decoder, long_edge, show_result, disable_cuda,
                    source, path_output_vid, path_output_json)
        

    # if cfg.top_down:
    #     print("top-down architecture")
    #     tracker_type=cfg.top_down.tracker_type
    #     detector_model=cfg.top_down.detector_model
    #     tracking_conf=cfg.top_down.tracking_conf
    #     verbose=cfg.top_down.verbose
    #     source=cfg.top_down.source
    #     gt=cfg.top_down.gt
    #     path_output_vid=cfg.top_down.path_output_vid
    #     output_json=cfg.top_down.output_json
    #     TopDown(tracker_type, detector_model, tracking_conf, verbose, source, gt, path_output_vid, output_json)
    # elif cfg.bottom_up:
    #     print("bottom_up architecture")
    #     if cfg.bottom_up=="pifpaf":
    #         print("using pifpaf Tracker")
    #         checkpoint=cfg.bottom_up.checkpoint
    #         decoder=cfg.bottom_up.decoder
    #         long_edge=cfg.bottom_up.long_edge
    #         show=cfg.bottom_up.show
    #         source=cfg.bottom_up.source
    #         video_output=cfg.bottom_up.video_output
    #         json_output=cfg.bottom_up.json_output
    #         disable_cuda=cfg.bottom_up.disable_cuda




if __name__ == "__main__":
    get_config()


    