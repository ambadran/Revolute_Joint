# MicroPython CAN Bus Library

MicroPython library for MCP2515, it works for most of the MicroPython boards.

With this library, you can,

- Send a CAN 2.0 frame
- Receive a CAN 2.0 frame
- Get data from OBD-II
- Set the masks and filters, there're 32 masks and filters.

## Getting Started

This MCP2515 library could be used for any MCU or platform supported by Micropython, such as Pyboard, ESP32, ESP8286, RP2040 etc.

### Wiring

If you are using a MCP2515 module with a MicroPython board, you can connect the SPI interface of your MicroPython board with MCP2515.

If you want to use an existing dev board with integrated CAN BUS support, please try:
- [CANBed RP2040](https://www.longan-labs.cc/1030014.html)

### Library installation

If using Thonny IDE, in the `Tools -> Manage Packages menu`, search for "MicroPython_CAN_BUS_MCP2515" and install it to the target board.

When using any other IDE or tools, you should install similarly.

### Example
```
'''
A simple exmaple to send data to can bus
'''

import sys
import time

from canbus import Can, CanError, CanMsg, CanMsgFlag

# setup
can = Can()

# initilize
ret = can.begin()
if ret != CanError.ERROR_OK:
    print("Error to initilize can!")
    sys.exit(1)
print("initlized succesufully!")

# send
while True:
    data = b"\x12\x34\x56\x78\x9A\xBC\xDE\xF0"
    
    # standard format frame message
    msg = CanMsg(can_id=0x123, data=data)
    error = can.send(msg)
    if error == CanError.ERROR_OK:
        print('1------------------------------')
            
    # extended format frame message
    msg = CanMsg(can_id=0x12345678, data=data, flags=CanMsgFlag.EFF)
    error = can.send(msg)
    if error == CanError.ERROR_OK:
        print('2------------------------------')

    time.sleep(1)
```

For more details of the Library API, please read docs/api.md for details.

# Contact us

Welcome to report issues to us.

If you have any other question, please feel free to contact support@longan-labs.cc


