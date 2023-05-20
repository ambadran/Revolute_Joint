from machine import Pin, I2C
from copy import deepcopy

original_setattr = deepcopy(setattr)

class AS5600:
    '''
    Instance to interface AS5600 module through micropython REPL
    '''
    ### AS5600 Module addresses ###
    address = { 'MODULE': 0x36, 'ZMCO': 0x00, 'ZPOS0': 0x01, 'ZPOS1': 0x02, 
            'MPOS0': 0x03, 'MPOS1': 0x04, 'MANG0': 0x05, 'MANG1': 0x06, 
            'CONF0': 0x07, 'CONF1': 0x08, 'RAW_ANGLE0': 0x0C, 'RAW_ANGLE1': 0x0D, 
            'ANGLE0': 0x0E, 'ANGLE1': 0x0F, 'AS5600_STATUS': 0x0B, 'AGC': 0x1A, 
            'MAGNITUDE0': 0x1B, 'MAGNITUDE1': 0x1C, 'BURN': 0xFF }


    def __init__(self):
        '''
        Initializing the module!
        '''
        self.i2c = I2C(1, sda=Pin(6), scl=Pin(7), freq=400_00)

    def read(self, register, num_bytes=1):
        '''
        :returns: byte_array
        :return: byte value from entered register to wanted number of bytes
        '''
        return self.i2c.readfrom_mem(self.address['MODULE'], register, num_bytes)

    def write(self, register, value):
        '''
        writes to wanted register the value argument
        '''
        return self.i2c.writeto_mem(self.address['MODULE'], register, chr(value))

    def __getattr__(self, address):
        '''
        reads the wanted address from the module and returns it
        '''
        try:
            return self.read(self.address[address])

        except KeyError:
            raise KeyError('Unknown address')

    def __setattr__(self, address, value):
        '''
        writes the wanted value to the address in the module
        '''
        try: 
            self.write(self.address[address], value)

        except KeyError:
            # not an address value, deal with it normally
            original_setattr(self, address, value)

        

as5600 = AS5600()
