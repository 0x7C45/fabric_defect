from machine import FPIOA
from machine import Pin
from time import sleep

fpioa = FPIOA()

LED_R = fpioa.set_function(62, FPIOA.GPIO62)
LED_G = fpioa.set_function(20, FPIOA.GPIO20)
LED_B = fpioa.set_function(63, FPIOA.GPIO63)
Button_USR = fpioa.set_function(53, FPIOA.GPIO53)

LED_R = Pin(62, Pin.OUT, pull = Pin.PULL_NONE, drive = 7)
LED_G = Pin(20, Pin.OUT, pull = Pin.PULL_NONE, drive = 7)
LED_B = Pin(63, Pin.OUT, pull = Pin.PULL_NONE, drive = 7)
Button_USR = Pin(53, Pin.IN, pull = Pin.PULL_DOWN)

LED_R.high()
LED_G.high()
LED_B.high()

color = 0 #0为R 1为G 2为B


while True:
    if color > 2:
        color = 0

    if Button_USR.value() == 1:
        if color == 0:
            LED_R.low()
            sleep(0.5)
            LED_R.high()
            color = color + 1
        elif color == 1:
            LED_G.low()
            sleep(0.5)
            LED_G.high()
            color = color + 1
        elif color == 2:
            LED_B.low()
            sleep(0.5)
            LED_B.high()
            color = color + 1
        else:
            color = 0

    else:
        LED_R.high()
        LED_G.high()
        LED_B.high()
