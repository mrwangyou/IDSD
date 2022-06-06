import os
import re
import subprocess

def isEmail(email):
    if re.match("^.+\\@(\\[?)[a-zA-Z0-9\\-\\.]+\\.([a-zA-Z]{2,3}|[0-9]{1,3})(\\]?)$", email) is not None:
    # if re.match("/^\w+@[a-z0-9]+\.[a-z]{2,4}$/", email) != None:
        return email
    else:
        raise Exception('Email Invalid!', email)

def isUrl(url):
    if re.match(r'^https?:/{2}\w.+$', url) is not None:
        return url
    else:
        raise Exception('URL Invalid!', url)

def isNumber(number, isnone=1):
    if isnone and number is None:
        return number
    try:
        if re.match(r'(^[-+]?([1-9][0-9]*|0)(\.[0-9]+)?$)', number):
            return number
    except:
        pass

    try:
        float(number)
        return number
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(number)
        return unicodedata.numeric(number)
    except (TypeError, ValueError):
        pass
 
    raise Exception('Number Invalid!', number)

def isLengthUnit(unit):
    if unit.upper() in ['FT', 'IN', 'M', 'KM']:
        return unit
    else:
        raise Exception('Length Unit Invalid!', unit)


def isAreaUnit(unit):
    if unit.upper() in ['M2', 'FT2']:
        return unit
    else:
        raise Exception('Area Unit Invalid!', unit)

def isVolumeUnit(unit):
    if unit.upper() in ['FT3', 'CC', 'M3', 'LTR']:
        return unit
    else:
        raise Exception('Volume Unit Invalid!', unit)

def isMassUnit(unit):
    if unit.upper() in ['LBS', 'KG']:
        return unit
    else:
        raise Exception('Mass and Weight Unit Invalid!', unit)

def isInertiaUnit(unit):
    if unit.upper() in ['SLUG*FT2', 'KG*M2']:
        return unit
    else:
        raise Exception('Moments of Inertia Unit Invalid!', unit)

def isAnglesUnit(unit):
    if unit.upper() in ['RAD', 'DEG']:
        return unit
    else:
        raise Exception('Angles Unit Invalid!', unit)

def isSpringUnit(unit):
    if unit.upper() in ['N/M', 'LBS/FT']:
        return unit
    else:
        raise Exception('Spring Force Unit Invalid!', unit)

def isDampingUnit(unit):
    if unit.upper() in ['N/M/SEC', 'LBS/FT/SEC']:
        return unit
    else:
        raise Exception('Damping Force Unit Invalid!', unit)

def isPowerUnit(unit):
    if unit.upper() in ['W', 'HP']:
        return unit
    else:
        raise Exception('Power Unit Invalid!', unit)

def isForceUnit(unit):
    if unit.upper() in ['LBS', 'N']:
        return unit
    else:
        raise Exception('Force Unit Invalid!', unit)
    
def isVelocityUnit(unit):
    if unit.upper() in ['KTS', 'FT/SEC', 'M/S']:
        return unit
    else:
        raise Exception('Velocity Unit Invalid!', unit)

def isTorqueUnit(unit):
    if unit.upper() in ['N*M', 'FT*LBS']:
        return unit
    else:
        raise Exception('Torque Unit Invalid!', unit)

def isPressureUnit(unit):
    if unit.upper() in ['PSF', 'PSI', 'ATM', 'PA', 'INHG']:
        return unit
    else:
        raise Exception('Pressure Unit Invalid!', unit)

def existAircraft(aircraft):
    if aircraft in os.listdir('./aircraft'):
        return aircraft
    else:
        raise Exception('Aircraft Missed!', aircraft)

def isBinary(variable:  bool):
    if type(variable) == bool or variable == None:
        return variable
    else:
        raise Exception('Binary Invalid!', variable)

def isCondition(condition):
    if condition is None:
        return condition
    for item in condition:
        if type(item) != str:
            raise Exception('Condition Type Invalid!', item, type(item))
    cdt_list = condition.split()
    if len(cdt_list) != 3:
        raise Exception('Condition Component Invalid!', condition)
    if not cdt_list[1].upper() in ['EQ', 'NE', 'LT', 'LE', 'GT', 'GE', '>', '>=', '==', '<=', '<', '!=']:
        raise Exception('Condition Operator Invalid!', condition, cdt_list[1].upper())
    # TBC

    return condition

def isType(typ:  "value|delta"):
    if typ is None or typ == "value" or typ == "delta":
        return typ
    else:
        raise Exception('Type Invalid!', typ, "value|delta")

def isAction(action:  "step|ramp|exp"):
    if action is None or action == 'step' or action == 'ramp' or action == 'exp':
        return action
    else:
        raise Exception('Action Invalid!', action, "step|ramp|exp")


if __name__ == '__main__':
    pass
