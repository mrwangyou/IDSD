from myModule import validation as val
import sys
print(sys.path)

dir_write = './code_wnn/output/script_test.xml'
indent_item = '  '

file = open(dir_write, 'w', encoding='UTF-8')


class Condition:
    def __init__(self, *text, subcon=None, logic="AND") -> None:
        if logic != "AND" and logic != "OR" and logic is not None:
            raise Exception('Logic Error', logic)
        
        self.text = text
        self.subcon = subcon
        self.logic = logic
    
    def output(self, tab=3):
        try:
            file.write(
                indent_item * tab + '<condition logic="{}">\n'.format(self.logic)
            )
            if self.text is not None:
                for item in self.text:
                    file.write(
                        indent_item * (tab + 1) + '{}\n'.format(item)
                    )
            if self.subcon is not None:
                self.subcon.output(tab + 1)
            file.write(
                indent_item * tab + '</condition>\n'
            )
        except:
            print('Condition Write Error!')
        

class Set:
    def __init__(self, name, value, typ=None, action=None, tc=None) -> None:
        if action is not None and tc is None:
            raise Exception(r"'tc' could not be set if 'action' is undefined.", 'action=' + str(action), 'tc=' + str(tc))

        self.name = name
        self.value = val.isNumber(value)
        self.typ = val.isType(typ)
        self.action = val.isAction(action)
        self.tc = val.isNumber(tc)
    
    def update(self, name=None, value=None, typ=None, action=None, tc=None):
        self.name = name if name is not None else self.name
        self.value = value if value is not None else self.value
        self.typ = typ if typ is not None else self.typ
        self.action = action if action is not None else self.action
        self.tc = tc if tc is not None else self.tc

    def output(self, tab=3):
        try:
            file.write(
                indent_item * tab + '<set name="{}" value="{}"'.format(self.name, self.value)
            )
            if self.typ is not None:
                file.write(
                    ' type="{}"'.format(self.typ)
                )
            if self.action is not None:
                file.write(
                    ' action="{}"'.format(self.action)
                )
                if self.tc is not None:
                    file.write(
                        ' tc="{}"'.format(self.tc)
                    )
            file.write('/>\n')
        except:
            print('Set Write Error!')


class Event:

    def __init__(self, condition, name=None, persistent=None, continuous=None, description=None, delay=None):
        if persistent == True and continuous == True:
            raise Exception('Persistent and continuous cannot both be set to true at the same time.')
        if type(condition) == Condition:
            self.condition = condition
        else:
            self.condition = Condition(condition)
        self.name = name
        self.persistent = val.isBinary(persistent)
        self.continuous = val.isBinary(continuous)
        self.description = str(description) if description is not None else None
        self.set = []
        self.delay = val.isNumber(delay)

    def update(self, name=None, url=None):
        self.name = name if not name is None else self.name
        self.url = val.isUrl(url) if not url is None else self.url
    
    def add_set(self, name, value, typ=None, action=None, tc=None):  # 可以有复数的内容采用单独的函数添加
        self.set.append(Set(name, value, typ, action, tc))
    
    def del_set(self):
        self.set.pop()

    def output(self, tab=2):
        try:
            file.write(
                indent_item * tab + '<event'.format(self.name)
            )
            if self.name is not None:
                file.write(
                    ' name="{}"'.format(self.name)
                )
            if self.persistent is not None:
                file.write(
                    ' persistent="{}"'.format(self.persistent)
                )
            if self.continuous is not None:
                file.write(
                    ' continuous="{}"'.format(self.continuous)
                )
            file.write('>\n')

            if self.description is not None:
                file.write(
                    indent_item * (tab + 1) + '<description> {} </description>\n'.format(self.description)
                )
            
            self.condition.output(tab + 1)

            for item in self.set:
                item.output(tab + 1)

            file.write(
                indent_item * tab + '</event>\n'
            )
        except:
            print('Event Write Error!')


class Script:
    """Class of a Script"""

    def __init__(self, name, description, aircraft, initialize, init_time, end_time, delta_time):
        self.name = name
        self.description = description
        self.aircraft = val.existAircraft(aircraft)
        self.initialize = initialize
        self.init_time = val.isNumber(init_time)
        self.end_time = val.isNumber(end_time)
        self.delta_time = val.isNumber(delta_time)
        self.event = []

    def add_event(self, event=None, name=None, value=None, typ=None, action=None, tc=None):  # 可以有复数的内容采用单独的函数添加
        if event is not None:
            self.event.append(event)
        else:
            if name is None or value is None:
                raise Exception("Name or value of an Event cannot be None!", name, value)
            else:
                self.event.append(Set(name, value, typ, action, tc))
    
    def del_event(self):
        self.event.pop()

    def output(self, tab=0):
        try:
            file.write(indent_item * tab + '<?xml version="1.0"?>\n')
            file.write(indent_item * tab + '<?xml-stylesheet type="text/xsl" href="http://jsbsim.sf.net/JSBSimScript.xsl"?>\n')
            file.write(indent_item * tab + '<runscript xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
            file.write(indent_item * (tab + 2) + 'xsi:noNamespaceSchemaLocation="http://jsbsim.sf.net/JSBSimScript.xsd"\n')
            file.write(indent_item * (tab + 2) + 'name="{}">\n'.format(self.name))
            file.write(indent_item * (tab + 1) + '<description>{}</description>\n'.format(self.description))
            file.write(indent_item * (tab + 1) + '<use aircraft="{}" initialize="{}"/>\n'.format(self.aircraft, self.initialize))
            file.write(indent_item * (tab + 1) + '<run start="{}" end="{}" dt="{}">\n'.format(self.init_time, self.end_time, self.delta_time))
            for item in self.event:
                item.output()
            file.write(indent_item * (tab + 1) + "</run>\n")
            file.write(indent_item * tab + "</runscript>\n")

        except:
            print('Script Write Error!')
    
    


if __name__ == '__main__':

    # C = Condition(
    #     "simulation/sim-time-sec  ge  0.5", 
    #     "simulation/sim-time-sec  ge  1.5", 
    #     subcon=Condition("simulation/sim-time-sec  ge  2.5")
    # )

    S = Script('F-16 aircraft test run', 'This run is for testing an F-16', 'f16', 'reset00', '0.0', 700, 0.00833333)

    E1 = Event(condition="simulation/sim-time-sec >= 0.25", name="starter")
    E1.add_set('propulsion/starter_cmd', 1)
    S.add_event(E1)

    E2 = Event(condition="propulsion/engine[0]/n2 >= 15")
    E2.add_set("propulsion/cutoff_cmd", 0)
    E2.add_set("fcs/throttle-cmd-norm[0]", 0.5, action="ramp", tc=2)
    S.add_event(E2)

    S.output()

    file.close()
