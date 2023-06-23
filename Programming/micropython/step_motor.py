from machine import Pin, PWM

class Motor:
    '''
    Instance to control a stepper motor driver through repl easily
    '''
    def __init__(self):
        self.step = PWM(Pin(14))
        self.step.freq(20)
        self.step.duty_u16(0)

        self.dir = Pin(15, Pin.OUT)
        self.dir.off()  # start as default cw value which is 0
        
        self.enable = Pin(13, Pin.OUT)
        # self.enable.off()
        self.enable.on()

        self.cw_value = False
        self.ccw_value = True

    def on(self):
        # self.enable.on()
        self.enable.off()
        self.step.duty_u16(32628)

    def off(self):
        # self.enable.off()
        self.enable.on()
        self.step.duty_u16(0)

    def freq(self, frequency):
        self.step.freq(frequency)

    def cw(self):
        self.dir.value(self.cw_value)

    def ccw(self):
        self.dir.value(self.ccw_value)

    @property
    def dir_value(self):
        return self.dir.value()

    def invert_default_dir_values(self):
        self.cw_value = not self.cw_value
        self.ccw_value = not self.ccw_value

        self.dir.toggle()


motor = Motor()
