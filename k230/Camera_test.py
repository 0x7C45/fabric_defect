import os, time, sys

from media.sensor import *
from media.display import *
from media.media import *

sensor_id = 2
sensor = None

try:
    sensor = Sensor(id=sensor_id, fps=60)

    #初始化
    sensor.reset()
    sensor.set_hmirror(True)
    sensor.set_vflip(True)

    #设置通道和输出格式
    sensor.set_framesize(sensor.VGA, chn=CAM_CHN_ID_0)
    sensor.set_pixformat(pix_format=sensor.RGB565, chn=CAM_CHN_ID_0)


    #初始化媒体，在IDE区域显示
    Display.init(Display.VIRT, width=640, height=480, to_ide=True)
    MediaManager.init()
    sensor.run()

    while True:
        os.exitpoint()

        img = sensor.snapshot(chn=CAM_CHN_ID_0)
        Display.show_image(img)

except KeyboardInterrupt as e:
    print("用户停止: ", e)
except BaseException as e:
    print(f"异常: {e}")

finally:
    if isinstance(sensor, Sensor):
        sensor.stop()

    Display.deinit()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)

    MediaManager.deinit()
