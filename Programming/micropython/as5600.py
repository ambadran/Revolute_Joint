from machine import Pin, I2C

i2c = I2C(1, sda=Pin(6), scl=Pin(7), freq=400_00)

class AS5600:
    '''
    Instance to interface AS5600 module through micropython REPL
    '''
    ### AS5600 Module addresses ###
    MODULE = 0x36
    registers = { 'ZMCO': 0x00, 'ZPOS0': 0x01, 'ZPOS1': 0x02, 
            'MPOS0': 0x03, 'MPOS1': 0x04, 'MANG0': 0x05, 'MANG1': 0x06, 
            'CONF0': 0x07, 'CONF1': 0x08, 'RAW_ANGLE0': 0x0C, 'RAW_ANGLE1': 0x0D, 
            'ANGLE0': 0x0E, 'ANGLE1': 0x0F, 'STATUS': 0x0B, 'AGC': 0x1A, 
            'MAGNITUDE0': 0x1B, 'MAGNITUDE1': 0x1C, 'BURN': 0xFF }

    bits = {'MH': (0x0B, 3), 'ML': (0x0B, 4), 'MD': (0x0B, 5)}

    double_regs = {'ZPOS': 0x01, 'MPOS': 0x03, 'MANG': 0x05, 'CONF': 0x07, 
            'RAW_ANGLE': 0x0C, 'ANGLE': 0x0E, 'MAGNITUDE': 0x1B }

    def __init__(self):
        '''
        Initializing the module!
        '''
        pass

    def read(self, register, num_bytes=1):
        '''
        :returns: byte_array
        :return: byte value from entered register to wanted number of bytes
        '''
        return i2c.readfrom_mem(self.MODULE, register, num_bytes)

    def write(self, register, value):
        '''
        writes to wanted register the value argument
        '''
        return i2c.writeto_mem(self.MODULE, register, chr(value))

    def __getattr__(self, address):
        '''
        reads the wanted address from the module and returns it
        '''
        if address in self.registers:
            return self.read(self.registers[address])[0]

        elif address in self.bits:
            # reg_val = self.read(self.bits[address][0])[0]
            # print(reg_val, 'reg_val')
            # print((1<< self.bits[address][1]), 'bit_checker')

            # bit_test = reg_val & (1<< self.bits[address][1])
            # print(bit_test, 'bit_test')

            # print(bit_test >> self.bits[address][1], 'final')
            
            return (self.read(self.bits[address][0])[0] & (1<<self.bits[address][1])) >> self.bits[address][1]

        elif address in self.double_regs:
            reg_val = self.read(self.double_regs[address], num_bytes=2)
            byte0 = reg_val[0]
            byte1 = reg_val[1]
            # double register values in as5600 are only 4bits from the second register
            byte1 &= 0b00001111  # filtering whatever value in the higher 4-bits
            byte_val = byte0<<8 | byte1

            return byte_val

        else:
            raise KeyError('Unknown address')

    def __setattr__(self, address, value):
        '''
        writes the wanted value to the address in the module
        '''
        if address in self.registers:
            self.write(self.registers[address], value)

        elif address in self.bits:
            reg_val = self.read(self.bits[address][0])[0]
            bit_order = self.bits[address][1]
            if value:
                reg_val |= (1<<bit_order)
                
            else:
                reg_val &= ~(1<<bit_order)

            self.write(self.bits[address][0], reg_val)

        elif address in self.double_regs:
            raise ValueError('Not implemented yet')

        else:
            raise KeyError('unknown address')

        

as5600 = AS5600()
