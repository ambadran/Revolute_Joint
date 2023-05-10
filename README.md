The Ultimate Revolute Joint could be reused for a multitude of applications.

Mechanics:
  - will be a stepper motor driving a 3d printed cycloidal drive

Electronics:
  - will be a MOSFET PIC controlled with a magnetometer for closed loop control

Programming:
  - will be in C for xc8 PIC which have input UART/I2C/SPI exact angle command and using a damped exact closed loop control algorithm will drive the mosfets using the correct sequence to get the revolute joint to wanted angle.
  - will have a python script to generate the cycloid for the mechanical design
  - will have a python script to easily communicate to the pic through a ttl converter or a micropython board using CLI and GUI to visualize

