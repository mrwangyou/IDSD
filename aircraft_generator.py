from myModule import validation as val

import time


dir_write = './output/aircraft_test.xml'
indent_item = ' '

file = open(dir_write, 'w', encoding='UTF-8')

class License:

    def __init__(self, name, url):
        self.name = name
        self.url = val.isUrl(url)
    
    # def get_name(self):
    #     return self.name

    # def get_url(self):
    #     return self.url

    def update(self, name=None, url=None):
        self.name = name or self.name
        self.url = val.isUrl(url) or self.url

    def print(self, tab=2):
        try:
            file.write(
                indent_item * tab + '<license>\n' +\
                indent_item * (tab + 1) + '<licenseName> {} </licenseName>\n'.format(self.name)  +\
                indent_item * (tab + 1) + '<licenseURL> {} </licenseURL>\n'.format(self.url) +\
                indent_item * tab + '</license>\n'
            )
        except:
            print('License write error!')


class Reference:

    def __init__(self, author, title, date='n/a', refID='None') -> None:
        self.author = author
        self.title = title
        self.date = date
        self.refID = refID

    
    def update(self, author=None, title=None, date=None, refID=None):
        self.author = author or self.author
        self.title = title or self.title
        self.date = date or self.date
        self.refID = refID or self.refID
        
    def print(self, tab=2):
        try:
            file.write(
                indent_item * tab + '<reference refID=\"{}\" author=\"{}\" title=\"{}\" date=\"{}\"/>\n'.format(self.refID, self.author, self.title, self.date)
            )
        except:
            print('Reference write error!')


class Fileheader:

    def __init__(self, 
                 author=None,
                 email=None,
                 organization=None,
                 file_creation_date=time.strftime("%Y-%m-%d", time.localtime()),
                 version=None,
                 lcs=None,
                 description=None,
                 note=None):
        self.author = author
        self.email = val.isEmail(email)
        self.organization = organization
        self.file_creation_date = file_creation_date
        self.version = version
        self.lcs = license
        self.description = description
        self.note = note
        self.reference = []
        self.A = []
    
    def update(self, 
               author=None,
               email=None,
               organization=None,
               file_creation_date=time.strftime("%Y-%m-%d", time.localtime()),
               version=None,
               lcs=None,
               description=None,
               note=None):
        self.A.append()
        self.author = author or self.author
        self.email = val.isEmail(email) or self.email
        self.organization = organization if not organization is None else self.organization
        self.file_creation_date = file_creation_date if not file_creation_date is None else self.file_creation_date
        self.version = version if not version is None else self.version
        self.lcs = lcs if not lcs is None else self.lcs
        self.description = description if not description is None else self.description
        self.note = note if not note is None else self.note

    def add_reference(self, author, title, date='n/a', refID='None'):  # 可以有复数的内容采用单独的函数添加
        self.reference.append(Reference(author, title, date, refID))
    
    def del_reference(self):
        self.reference.pop()

    def print(self, tab=1):
        try:
            file.write(indent_item * tab + '<fileheader>\n')

            if not self.author is None:
                file.write(
                    indent_item * (tab + 1) + '<author> {} </author>\n'.format(self.author)
                )

            if not self.email is None:
                file.write(
                    indent_item * (tab + 1) + '<email> {} </email>\n'.format(self.email)
                )

            if not self.organization is None:
                file.write(
                    indent_item * (tab + 1) + '<organization> {} </organization>\n'.format(self.organization)
                )

            if not self.file_creation_date is None:
                file.write(
                    indent_item * (tab + 1) + '<fileCreationDate> {} </fileCreationDate>\n'.format(self.file_creation_date)
                )

            if not self.version is None:
                file.write(
                    indent_item * (tab + 1) + '<version> {} </version>\n'.format(self.version)
                )

            if not self.license is None:
                self.license.print(tab + 1)
            
            if not self.description is None:
                file.write(
                    indent_item * (tab + 1) + '<description> {} </description>\n'.format(self.description)
                )            

            if not self.note is None:
                file.write(
                    indent_item * (tab + 1) + '<note> {} </note>\n'.format(self.note)
                )

            if not self.reference is None:
                for item in self.reference:
                    item.print(tab + 1)
            
            file.write(indent_item * tab + '</fileheader>\n')
        except:
            print('Fileheader write error!')


class Location:

    def __init__(self, name, unit, x, y, z):
        self.name = name
        self.unit = val.isLengthUnit(unit)
        self.x = str(val.isNumber(x))
        self.y = str(val.isNumber(y))
        self.z = str(val.isNumber(z))

    def update(self, name=None, unit=None, x=None, y=None, z=None):
        self.name = name if not name is None else self.name
        self.unit = val.isLengthUnit(unit) if not unit is None else self.unit
        self.x = str(val.isNumber(x)) if not x is None else self.x
        self.y = str(val.isNumber(y)) if not y is None else self.y
        self.z = str(val.isNumber(z)) if not z is None else self.z

    def print(self, tab=2):
        try:
            file.write(
                indent_item * tab + '<location name=\"{}\" unit=\"{}\">\n'.format(self.name, self.unit) +\
                indent_item * (tab + 1) + '<x> {} </x>\n'.format(self.x) +\
                indent_item * (tab + 1) + '<y> {} </y>\n'.format(self.y) +\
                indent_item * (tab + 1) + '<z> {} </z>\n'.format(self.z) +\
                indent_item * tab + '</location>\n'
            )
        except:
            print('Location write error!')


class Metrics:

    def __init__(self, 
                 wingarea,
                 wingspan,
                 chord,
                 htailarea,
                 htailarm,
                 vtailarea,
                 vtailarm):
        self.wingarea = str(wingarea)
        self.wingspan = str(wingspan)
        self.chord = str(chord)
        self.htailarea = str(htailarea)
        self.htailarm = str(htailarm)
        self.vtailarea = str(vtailarea)
        self.vtailarm = str(vtailarm)
        self.location = []
    
    def update(self, 
               wingarea=None,
               wingspan=None,
               chord=None,
               htailarea=None,
               htailarm=None,
               vtailarea=None,
               vtailarm=None):
        self.wingarea = wingarea if not wingarea is None else self.wingarea
        self.wingspan = wingspan if not wingspan is None else self.wingspan
        self.chord = chord if not chord is None else self.chord
        self.htailarea = htailarea if not htailarea is None else self.htailarea
        self.htailarm = htailarm if not htailarm is None else self.htailarm
        self.vtailarea = vtailarea if not vtailarea is None else self.vtailarea
        self.vtailarm = vtailarm if not vtailarm is None else self.vtailarm

    def add_location(self, name, unit, x, y, z):
        self.location.append(Location(name, val.isLengthUnit(unit), val.isNumber(x), val.isNumber(y), val.isNumber(z)))
    
    def del_location(self):
        self.location.pop()

    def print(self, tab=1):
        try:
            file.write(indent_item * tab + '<metrics>\n')
            file.write(indent_item * (tab + 1) + '<wingarea unit="FT2"> {} </wingarea>\n'.format(self.wingarea))
            file.write(indent_item * (tab + 1) + '<wingspan unit="FT2"> {} </wingspan>\n'.format(self.wingspan))
            file.write(indent_item * (tab + 1) + '<chord unit="FT2"> {} </chord>\n'.format(self.chord))
            file.write(indent_item * (tab + 1) + '<htailarea unit="FT2"> {} </htailarea>\n'.format(self.htailarea))
            file.write(indent_item * (tab + 1) + '<htailarm unit="FT2"> {} </htailarm>\n'.format(self.htailarm))
            file.write(indent_item * (tab + 1) + '<vtailarea unit="FT2"> {} </vtailarea>\n'.format(self.vtailarea))
            file.write(indent_item * (tab + 1) + '<vtailarm unit="FT2"> {} </vtailarm>\n'.format(self.vtailarm))
            for item in self.location:
                item.print(tab + 1)
            file.write(indent_item * tab + '</metrics>\n')
        except:
            print('Metrics write error!')


class Pointmass:

    def __init__(self, name, weight, location):
        self.name = name
        self.weight = val.isNumber(weight)
        self.location = location
    
    def update(self, 
               name=None,
               weight=None,
               location=None):
        self.name = name if not name is None else self.name
        self.weight = val.isNumber(weight) if not weight is None else self.weight
        self.location = location if not location is None else self.location

    def print(self, tab=2):
        try:
            file.write(indent_item * tab + '<pointmass name=\"{}\">\n'.format(self.name))
            file.write(indent_item * (tab + 1) + '<weight unit=\"LBS\"> {} </weight>\n'.format(self.weight))
            self.location.print(tab + 1)
            file.write(indent_item * tab + '</pointmass>\n')
        except:
            print('Pointmass write error!')


class MassBalance:

    def __init__(self, ixx, iyy, izz, ixy, ixz, iyz, emptywt, location, pointmass) -> None:
        self.ixx = str(val.isNumber(ixx))
        self.iyy = str(val.isNumber(iyy))
        self.izz = str(val.isNumber(izz))
        self.ixy = str(val.isNumber(ixy))
        self.ixz = str(val.isNumber(ixz))
        self.iyz = str(val.isNumber(iyz))
        self.emptywt = str(val.isNumber(emptywt))
        self.location = location
        self.pointmass = pointmass
    
    def update(self, 
               ixx=None, 
               iyy=None, 
               izz=None, 
               ixy=None, 
               ixz=None, 
               iyz=None, 
               emptywt=None, 
               location=None, 
               pointmass=None):
        self.ixx = str(val.isNumber(ixx)) if not ixx is None else self.ixx
        self.iyy = str(val.isNumber(iyy)) if not iyy is None else self.iyy
        self.izz = str(val.isNumber(izz)) if not izz is None else self.izz
        self.ixy = str(val.isNumber(ixy)) if not ixy is None else self.ixy
        self.ixz = str(val.isNumber(ixz)) if not ixz is None else self.ixz
        self.iyz = str(val.isNumber(iyz)) if not iyz is None else self.iyz
        self.emptywt = str(val.isNumber(emptywt)) if not emptywt is None else self.emptywt
        self.location = location if not location is None else self.location
        self.pointmass = pointmass if not pointmass is None else self.pointmass

    def print(self, tab=1):
        try:
            file.write(indent_item * tab + '<mass_balance negated_crossproduct_inertia=\"true\">\n')
            file.write(indent_item * (tab + 1) + '<ixx unit="SLUG*FT2"> {} </ixx>\n'.format(self.ixx))
            file.write(indent_item * (tab + 1) + '<iyy unit="SLUG*FT2"> {} </iyy>\n'.format(self.iyy))
            file.write(indent_item * (tab + 1) + '<izz unit="SLUG*FT2"> {} </izz>\n'.format(self.izz))
            file.write(indent_item * (tab + 1) + '<ixy unit="SLUG*FT2"> {} </ixy>\n'.format(self.ixy))
            file.write(indent_item * (tab + 1) + '<ixz unit="SLUG*FT2"> {} </ixz>\n'.format(self.ixz))
            file.write(indent_item * (tab + 1) + '<iyz unit="SLUG*FT2"> {} </iyz>\n'.format(self.iyz))
            file.write(indent_item * (tab + 1) + '<emptywt unit="LBS"> {} </emptywt>\n'.format(self.emptywt))
            self.location.print(tab + 1)
            self.pointmass.print(tab + 1)
            file.write(indent_item * tab + '</mass_balance>\n')
        except:
            print('Mass_balance write error!')


class Aircraft:
    """Class of an aircraft"""
    def __init__(self, name, version, release, fileheader, metrics, massBalance):
        self.name = name
        self.version = version
        self.release = release
        self.fileheader = fileheader
        self.metrics = metrics
        self.massBalance = massBalance
    
    def print(self, tab=0):
        try:
            file.write(indent_item * tab + '<?xml version="1.0"?>\n')
            file.write(indent_item * tab + '<?xml-stylesheet type="text/xsl" href="http://jsbsim.sourceforge.net/JSBSim.xsl"?>\n')
            file.write(indent_item * tab + '<fdm_config name="{}" version="{}" release="{}"\n'.format(self.name, self.version, self.release))
            file.write(indent_item * (tab + 1) + 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
            file.write(indent_item * (tab + 1) + 'xsi:noNamespaceSchemaLocation="http://jsbsim.sourceforge.net/JSBSim.xsd">\n\n')
            self.fileheader.print(tab + 1)
            file.write('\n')
            self.metrics.print(tab + 1)
            file.write('\n')
            self.massBalance.print(tab + 1)
            file.write('\n' + indent_item * tab + '</fdm_config>\n')
        except:
            print('Aircraft write error!')


if __name__ == '__main__':

    L = License('licenseName', 'https://www.baidu.html')

    F = Fileheader(author='Gustum', email='wnn@qq.com', organization='Jiaotong', version='1.0', license=L, note='Wow!')
    F.add_reference('wnn', 'Titl', refID='4')
    F.add_reference('wnn2', 'ti2', '2022/2/23')

    M = Metrics(174, 35.8, 4.9, 21.9, 15.7, 16.5, 15.7)
    M.add_location('AERORP', 'IN', 43.2, 0.0, 59.4)
    M.add_location('EVEPOINT', 'IN', 37.0, 0.0, 48.0)
    M.add_location('VRP', 'IN', 42.6, 0.0, 38.5)

    L_CG = Location('CG', 'IN', -193, 0, -5.1)
    L_PILOT = Location('POINTMASS', 'IN', -336.2, 0, 0)
    PM = Pointmass('Pilot', 230, L_PILOT)


    MB = MassBalance(9496, 55814, 63100, 0, -982, 0, 17400, L_CG, PM)

    A = Aircraft('General Dynamics F-16A', '2.0', 'PRODUCTION', F, M, MB)
    A.print()

    file.close()
