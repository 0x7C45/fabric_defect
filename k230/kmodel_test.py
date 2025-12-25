from libs.PipeLine import PipeLine, ScopedTiming
from libs.YOLO import YOLOv8
import os,sys,gc
import ulab.numpy as np
import image

if __name__=="__main__":
    # 显示模式，默认"hdmi",可以选择"hdmi"和"lcd"
    display_mode="hdmi"
    rgb888p_size=[1280,720]
    if display_mode=="hdmi":
        display_size=[1920,1080]
    else:
        display_size=[800,480]
    # 可以根据您的模型自行修改路径参数
    kmodel_path="/data/best.kmodel"
    labels = ['no_defect', 'hole', 'stain', 'three_strands', 'knot', 'flower_jump',
              'hundred_feet', 'hair_balls', 'coarse_warp', 'loose_warp', 'broken_warp',
              'hanging_warp', 'coarse_weft', 'weft_shrink', 'starch_spot', 'warp_knot',
              'star_jump', 'broken_spandex', 'uneven_density', 'surface_defect', 'weave_defect']
    confidence_threshold = 0.8
    model_input_size=[224,224]
    # 初始化PipeLine
    pl=PipeLine(rgb888p_size=rgb888p_size,display_size=display_size,display_mode=display_mode)
    pl.create()
    # 初始化YOLOv8实例
    yolo=YOLOv8(task_type="detect",mode="video",kmodel_path=kmodel_path,labels=labels,rgb888p_size=rgb888p_size,model_input_size=model_input_size,display_size=display_size,conf_thresh=confidence_threshold,debug_mode=0)
    yolo.config_preprocess()
    try:
        while True:
            os.exitpoint()
            with ScopedTiming("total",1):
                # 逐帧推理
                img=pl.get_frame()
                res=yolo.run(img)
                yolo.draw_result(res,pl.osd_img)
                pl.show_image()
                gc.collect()
    except Exception as e:
        sys.print_exception(e)
    finally:
        yolo.deinit()
        pl.destroy()
