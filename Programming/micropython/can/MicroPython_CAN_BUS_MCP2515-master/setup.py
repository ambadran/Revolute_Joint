from setuptools import setup, find_packages

__project__ = 'MicroPython_CAN_BUS_MCP2515'
# __packages__ = ['canbus']
__packages__ = find_packages()
# __package_dir__ = {'canbus': 'samples'}
__desc__ = 'A beginner-friendly CAN BUS library (MCP2515), supported various MicroPython boards.'
__version__ = '0.0.1'
__author__ = "Longan Labs"
__author_email__ = 'info@longan-labs.cc'
__license__ = 'MIT'
__url__ = 'https://github.com/Longan-Labs/MicroPython_CAN_BUS_MCP2515'
__keywords__ = [
    'raspberry',
    'pi',
    'pico',
    'electronics',
]
__classifiers__ = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Customer Service',
        'Intended Audience :: Manufacturing',
        'Programming Language :: Python :: Implementation :: MicroPython',
    ]
__long_description__ = """A beginner-friendly CAN BUS library (MCP2515), supported various MicroPython boards.

How to import the library
```python
from canbus import Can, CanError, CanMsg, CanMsgFlag
```

Refer to more details at [github](https://github.com/Longan-Labs/MicroPython_CAN_BUS_MCP2515).
"""

setup(
    name=__project__,
    version=__version__,
    description=__desc__,
    long_description=__long_description__,
    long_description_content_type='text/markdown',
    url=__url__,
    author=__author__,
    author_email=__author_email__,
    license=__license__,
    classifiers=__classifiers__,
    keywords=__keywords__,
    packages=__packages__,
    # package_dir=__package_dir__,
)