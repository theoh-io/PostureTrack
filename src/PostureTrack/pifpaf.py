import openpifpaf

#import argparse
import json
import logging
import os
import time

import torch

from openpifpaf import decoder, logger, network, show, visualizer, __version__
from openpifpaf.predictor import Predictor
from openpifpaf.stream import Stream

import argparse
from argparse import Namespace

import hydra
from omegaconf import DictConfig, OmegaConf

LOG = logging.getLogger(__name__)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():  # pylint: disable=too-many-statements,too-many-branches
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.video',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.Factory.cli(parser)
    decoder.cli(parser)
    logger.cli(parser)
    Predictor.cli(parser)
    show.cli(parser)
    Stream.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('--source', default='0',
                        help=('OpenCV source url. Integer for webcams. '
                              'Or ipwebcam urls (rtsp/rtmp). '
                              'Use "screen" for screen grabs.'))
    parser.add_argument('--video-output', default=None, nargs='?', const=True,
                        help='video output file or "virtualcam"')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='json output file')
    parser.add_argument('--separate-debug-ax', default=False, action='store_true')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    logger.configure(args, LOG)  # logger first

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
             args.device, torch.cuda.is_available(), torch.cuda.device_count())

    # decoder.configure(args)
    # network.Factory.configure(args)
    # Predictor.configure(args)
    # show.configure(args)
    # Stream.configure(args)
    # visualizer.configure(args)

    # check whether source should be an int
    # if lenargs.source)) == 1:
    #     args.source = int(args.source)

    # standard filenames
    # if args.video_output is True:
    #     args.video_output = '{}.openpifpaf.mp4'.format(args.source)
    #     if os.path.exists(args.video_output):
    #         os.remove(args.video_output)
    # assert args.video_output is None or not os.path.exists(args.video_output)
    # if args.json_output is True:
    #     args.json_output = '{}.openpifpaf.json'.format(args.source)
    #     if os.path.exists(args.json_output):
    #         os.remove(args.json_output)
    # assert args.json_output is None or not os.path.exists(args.json_output)

    return args

def cli2(args):    
    # setattr(args, "checkpoint", "tshufflenetv2k30")
    # #setattr(args, "decoder", "trackingpose:0")
    # args.decoder=['trackingpose:0']   
    # #args.decoder=None
    # setattr(args, "long_edge", 321)
    # setattr(args, "show", True)
    # setattr(args, "source", "../../skating.mp4")
    # #setattr(args, "video_output", "skating")
    # #setattr(args, "json_output", "skating")
    # #setattr(args, "disable_cuda", "False")
    # device = torch.device('cpu')
    # disable_cuda=False
    # if not disable_cuda and torch.cuda.is_available():
    #     device = torch.device('cuda')
    # setattr(args, "device", device)

    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args)
    show.configure(args)
    Stream.configure(args)
    visualizer.configure(args)

    #check whether source should be an int
    # if len(args.source)) == 1:
    #     args.source = int(args.source)

    #standard filenames
    if args.video_output is True:
        args.video_output = '{}.openpifpaf.mp4'.format(args.source)
        if os.path.exists(args.video_output):
            os.remove(args.video_output)
    assert args.video_output is None or not os.path.exists(args.video_output)
    if args.json_output is True:
        args.json_output = '{}.openpifpaf.json'.format(args.source)
        if os.path.exists(args.json_output):
            os.remove(args.json_output)
    assert args.json_output is None or not os.path.exists(args.json_output)

    return args


@hydra.main(version_base=None, config_path="config", config_name="cfg_pifpaf")
def hydra_pifpaf(cfg: DictConfig) -> None:
    global args
    #print(OmegaConf.to_yaml(cfg))
    args.checkpoint=cfg.settings.checkpoint
    args.decoder=cfg.settings.decoder  
    args.long_edge=cfg.settings.long_edge
    args.show=cfg.settings.show
    args.source = cfg.settings.source
    args.video_output=cfg.settings.video_output
    args.json_output=cfg.settings.json_output
    disable_cuda=cfg.settings.disable_cuda
    device = torch.device('cpu')
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    args.device=device




def PifPaf(checkpoint, decoder, long_edge, show_result, disable_cuda,
                    source, path_output_vid, path_output_json):
    global args
    args = cli()
    #hydra_pifpaf()
    #adding to ags arguments from cfg
    args.checkpoint=checkpoint
    args.decoder=decoder  
    args.long_edge=long_edge
    args.show=show_result
    args.source = source
    args.video_output=path_output_vid
    args.json_output=path_output_json
    disable_cuda=disable_cuda
    device = torch.device('cpu')
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    args.device=device

    args=cli2(args)
    print(args.decoder)

    Predictor.loader_workers = 1
    predictor = Predictor(
        visualize_image=(not args.json_output or args.video_output),
        visualize_processed_image=args.debug,
    )
    capture = Stream(args.source, preprocess=predictor.preprocess)

    annotation_painter = show.AnnotationPainter()
    animation = show.AnimationFrame(
        video_output=args.video_output,
        second_visual=args.separate_debug_ax,
    )
    if not args.json_output or args.video_output:
        ax, ax_second = animation.frame_init()
        visualizer.Base.common_ax = ax_second if args.separate_debug_ax else ax

    last_loop = time.perf_counter()
    for (ax, ax_second), (preds, _, meta) in \
            zip(animation.iter(), predictor.dataset(capture)):
        start_post = time.perf_counter()
        if args.json_output:
            with open(args.json_output, 'a+') as f:
                json.dump({
                    'frame': meta['frame_i'],
                    'predictions': [ann.json_data() for ann in preds]
                }, f, separators=(',', ':'))
                f.write('\n')
        if (not args.json_output or args.video_output) \
           and (args.separate_debug_ax or not args.debug_indices):
            image = visualizer.Base.image()
            ax.imshow(image)
            annotation_painter.annotations(ax, preds)
        postprocessing_time = time.perf_counter() - start_post
        if animation.last_draw_time is not None:
            postprocessing_time += animation.last_draw_time

        LOG.info('frame %d, loop time = %.0fms (pre = %.1fms, post = %.1fms), FPS = %.1f',
                 meta['frame_i'],
                 (time.perf_counter() - last_loop) * 1000.0,
                 meta['preprocessing_s'] * 1000.0,
                 postprocessing_time * 1000.0,
                 1.0 / (time.perf_counter() - last_loop))
        last_loop = time.perf_counter()


if __name__ == '__main__':
    main()
