from machine import Pin, PWM
from micropython import const


class Motor:
    '''
    Instance to control a stepper motor driver through repl easily
    '''
    CW_STATE = 1
    CCW_STATE = 0
    def __init__(self):
        self.step = PWM(Pin(14))
        self.step.freq(20)
        self.step.duty_u16(0)

        self.dir_pin = Pin(15, Pin.OUT)
        self.dir_pin.off()  # start as default cw value which is 0
        
        self.enable = Pin(13, Pin.OUT)
        # self.enable.off()
        self.enable.on()

    def on(self):
        # self.enable.on()
        self.enable.off()
        self.step.duty_u16(32628)

    def off(self):
        # self.enable.off()
        self.enable.on()
        self.step.duty_u16(0)

    def unhold(self):
        self.enable.off()
        self.step.duty_u16(32628)

    def hold(self):
        self.enable.off()
        self.step.duty_u16(0)

    def freq(self, frequency):
        self.step.freq(frequency)

    def cw(self):
        self.dir_pin.value(self.CW_STATE)

    def ccw(self):
        self.dir_pin.value(self.CCW_STATE)

    @property
    def direction(self):
        return self.dir_pin.value()

    @direction.setter
    def direction(self, value):
        self.dir_pin.value(value)

    def toggle(self):
        self.dir_pin.toggle()

    def invert_default_dir_values(self):
        self.CW_STATE = self.CW_STATE ^ 1
        self.CCW_STATE = self.CCW_STATE ^ 1

        self.toggle()


motor = Motor()
