from machine import Pin
from machine import FPIOA

import time

fpioa = FPIOA()
#设置引脚功能
fpioa.set_function(62, FPIOA.GPIO62)
fpioa.set_function(20, FPIOA.GPIO20)
fpioa.set_function(63, FPIOA.GPIO63)
#初始化引脚，这里采用实例化方式
pinR = Pin(62, Pin.OUT, pull=Pin.PULL_NONE, drive = 7)
pinG = Pin(20, Pin.OUT, pull=Pin.PULL_NONE, drive = 7)
pinB = Pin(63, Pin.OUT, pull=Pin.PULL_NONE, drive = 7)
#高电平熄灭
pinR.high()
pinG.high()
pinB.high()
#RGB轮流亮起
while True:
    pinB.high()
    pinR.low()
    time.sleep(0.5)

    pinR.high()
    pinG.low()
    time.sleep(0.5)

    pinG.high()
    pinB.low()
    time.sleep(0.5)
