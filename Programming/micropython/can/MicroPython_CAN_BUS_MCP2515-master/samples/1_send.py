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
