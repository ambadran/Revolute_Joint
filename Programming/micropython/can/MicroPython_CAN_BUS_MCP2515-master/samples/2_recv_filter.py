'''
A simple exmaple to recv data from can bus, and apply filter settings
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

# configure masks & filters (optional)

## recv only standard format frame with id 0x123
#mask=0x3FF
#filter_can_id=0x123
#is_ext = False

## recv only extended format frame with id 0x12345678
mask=0x1FFFFFFF
filter_can_id=0x12345678
is_ext = True

## set1
can.init_mask(0, is_ext, mask)
can.init_filter(0, is_ext, filter_can_id)
can.init_filter(1, is_ext, filter_can_id)
## set2
can.init_mask(1, is_ext, mask)
can.init_filter(2, is_ext, filter_can_id)
can.init_filter(3, is_ext, filter_can_id)
can.init_filter(4, is_ext, filter_can_id)
can.init_filter(5, is_ext, filter_can_id)
print('masks&filtes configured')

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
