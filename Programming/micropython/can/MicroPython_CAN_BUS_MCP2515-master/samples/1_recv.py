'''
A simple exmaple to recv data from can bus
'''

import sys
import time

from canbus import Can, CanError

# setup
can = Can()

# initilize
ret = can.begin()
if ret != CanError.ERROR_OK:
    print("Error to initilize can!")
    sys.exit(1)
print("initlized succesufully!")

# receive
while True:
    if can.checkReceive():
        error, msg = can.recv()
        if error == CanError.ERROR_OK:
            print('------------------------------')
            print("can id: %#x" % msg.can_id)
            print("is rtr frame:", msg.is_remote_frame)
            print("is eff frame:", msg.is_extended_id)
            print("can data hex:", msg.data.hex())
            print("can data dlc:", msg.dlc)
    else:
        time.sleep(0.1)
