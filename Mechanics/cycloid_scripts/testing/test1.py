import ezdxf

doc = ezdxf.new('R2010')
doc.units = ezdxf.units.MM

class MSP_Wrapper:  # the return object of doc.modelspace
    def __init__(self, doc: ezdxf.document.Drawing):
        self.msp = doc.modelspace()


msp_wrapper = MSP_Wrapper(doc)

msp_wrapper.msp.add_circle([0, 0, 0], 5)

doc.saveas('test.dxf')

