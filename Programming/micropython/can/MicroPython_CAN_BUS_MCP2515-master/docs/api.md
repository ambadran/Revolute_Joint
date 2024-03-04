# Library import
```python
from canbus import Can, CanError, CanMsg, CanMsgFlag
```

# API List

## Main functional classes

### Can class

- Can()

  Create a can bus instance and return it.

  Example:
  ```python
  can = Can()
  ```

- can.begin()

  Initialize the can bus and return a CanError value.

  Example:
  ```python
  error = can.begin()
  if error != CanError.ERROR_OK:
      print("Failed to initialize can!")
  ```

- can.checkReceive()

  Check if there is any message available and return True/False.

- can.recv()

  Read a message from the can bus and return a CanError value and a CanMsg object.

  Example:
  ```python
  if can.checkReceive():
      error, msg = can.recv()
      if error == CanError.ERROR_OK:
          print("Can id", msg.can_id)
          print("Can data", msg.data)
  ```

- can.send(msg)

  Write a message to the can bus and return a CanError value.

  Example:
  ```python
  msg = CanMsg(can_id=0x123, data=b"\x12\x34\x56\x78\x9A\xBC\xDE\xF0")
  error = can.send(msg)
  if error != CanError.ERROR_OK:
      print("Failed to send!")
  ```

### CanMsg class

- CanMsg()

  Create a can bus message to send to the can bus.

  Example:
  ```python
  data = b"\x12\x34\x56\x78\x9A\xBC\xDE\xF0"
  
  # Send a standard format frame message
  msg = CanMsg(can_id=0x123, data=data)
  can.send(msg)

  # Send an extended format frame message
  msg = CanMsg(can_id=0x12345678, data=data, flags=CanMsgFlag.EFF)
  can.send(msg)

  # Send a remote transmission request frame (usually no data) with extended id
  msg = CanMsg(can_id=0x12345678, flags=CanMsgFlag.EFF | CanMsgFlag.RTR )
  can.send(msg)
  ```

- Attributes

  When receiving a frame, we can get the message details from the following attributes:

    - can_id: the 11-bit or 29-bit can id
    - is_remote_frame: whether it is a remote transmission request message
    - is_extended_id: whether it is an extended format frame message
    - data: the message data
    - dlc: the length of the message data

    Example:
    ```python
    if can.checkReceive():
        error, msg = can.recv()
        if error == CanError.ERROR_OK:
            print("Can id", msg.can_id)
            print("Is rtr frame", msg.is_remote_frame)
            print("Is eff frame", msg.is_extended_id)
            print("Can data", msg.data)
            print("Can data dlc", msg.dlc)
    ```

## Utility classes

### CanMsgFlag class

The possible values are:

- CanMsgFlag.RTR
- CanMsgFlag.EFF

These flags indicate whether a message is a remote transmission request or an extended format frame.

### CanError class

Many functions will return a CanError value. The possible values are:

- CanError.ERROR_OK
- CanError.ERROR_FAIL
- ...

Note: any value other than CanError.ERROR_OK means an actual error.

## advanced usages

### filters & masks

- init masks

  set the mask register of MCP2515, the function declaration is as below:

  ```
  error = init_mask(self, mask, is_ext_id, mask_can_id)
  ```

  arguments:

    - mask: which register to set, values can only be 0 or 1
    - is_ext_id: whether it is set for extended format frame id or standard format frame id.
    - mask_can_id: 11 bit or 29 bit can id mask accordingly

  The function will return a CanError value.

  Example:
  ```python
  error = can.init_mask(0, False, 0x3FF)
  error = can.init_mask(1, True, 0x1FFFFFFF)
  ```

- init filters

  set the filter register of MCP2515, the function declaration is as below:

  ```
  error = init_filter(self, filter, is_ext_id, filter_can_id)
  ```

  arguments:

    - filter: which register to set, values can be from 0 to 5.
    - is_ext_id: whether it is set for extended format frame id or standard format frame id.
    - filter_can_id: 11 bit or 29 bit can id filter accordingly

  The function will return a CanError value.

  Example:
  ```python
  error = can.init_filter(0, False, 0x123)
  error = can.init_filter(1, True, 0x12345678)
  ```

- Notes

  there are 2 masks and 6 filters on MCP2515.