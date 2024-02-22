class Pin:
    def __init__(a, b):
        pass

class Motor:
    '''
    H-Bridge controller for DC motors

    H-bridge is two pnp and two npn
    NOTE: pnp is active high, npn is active low
    '''
    # class dictionary to map cw_ccw value to corresponding activating cw/ccw method

    def __init__(self, p1_pin_num: int, p2_pin_num: int, n1_pin_num: int, n2_pin_num: int):
        '''
        Constructor
        '''
        # Motor control pins
        self.p1 = Pin(p1_pin_num, Pin.OUT)
        self.p2 = Pin(p2_pin_num, Pin.OUT)
        self.n1 = Pin(n1_pin_num, Pin.OUT)
        self.n2 = Pin(n2_pin_num, Pin.OUT)

        # Initializing with motor off
        self.off()

        # private attributes
        self._cw_ccw: bool = True  # True for clockwise, False for anti-clockwise
        self.cw_values = {True: self.cw, False: self.ccw}

    def on(self):
        '''
        activates H-bridge to the last known cw_ccw value
        '''
        self.direction(self._cw_ccw)

    def off(self):
        '''
        deactivates H-bridge
        '''
        self.p1.off()
        self.p2.off()
        self.n1.on()
        self.n2.on()

    def cw(self):
        '''
        clockwise motion
        '''
        # first closing the circuit to ensure no dead-time short
        self.n1.on()
        self.n2.on()

        # openning n1 and p2
        self.p1.on()
        self.p2.off()
        self.n1.on()
        self.n2.off()

    def ccw(self):
        '''
        anti-clockwise motion
        '''
        # first closing the circuit to ensure no dead-time short
        self.n1.on()
        self.n2.on()

        # openning n2 and p1
        self.p1.off()
        self.p2.on()
        self.n1.off()
        self.n2.on()


    @property
    def direction(self):
        '''
        returns current cw_ccw
        '''
        return self._cw_ccw

    @direction.setter
    def direction(self, value: bool):
        '''
        sets motor direction
        '''
        self.cw_values[value]()
        self.cw_ccw = value





