#
# TI SimpleLink SensorTag 2015
# Date: 2015 07 07
#
# Sensor: Gyro, Accelerometer and Magnetometer
# Values: 9 bytes x, y, z for each sensor
# Note: Sensor values have not been validated
#
import struct, sys, traceback, time
from bluepy.btle import UUID, Peripheral, BTLEException

def TI_UUID(val):
    return UUID("%08X-0451-4000-b000-000000000000" % (0xF0000000+val))

config_uuid = TI_UUID(0xAA82)
data_uuid = TI_UUID(0xAA81)

# Bit settings to turn on individual movement sensors
# bits 0 - 2: Gyro x, y z
# bits 3 - 5: Accelerometer x, y, z
# bit: 6: Magnetometer turns on X, y , z with one bit

#gyroOn = 0x0700
#accOn = 0x3800
#magOn = 0xC001

#sensorOnVal = gyroOn | magOn | accOn
sensorOnVal = 0x7F02

sensorOn = struct.pack("BB", (sensorOnVal >> 8) & 0xFF, sensorOnVal & 0xFF)
sensorOff = struct.pack("BB", 0x00, 0x00)

if len(sys.argv) != 3:
  print "Fatal, must pass label and time in seconds: <label> <time>"
  quit()

sensorLeft = '24:71:89:BB:FA:00'
sensorRight = '24:71:89:C0:BC:84'
label = sys.argv[1]
time_read = float(sys.argv[2])

try:
  print "Info, trying to connect to:", sensorLeft
  l = Peripheral(sensorLeft)
  print "Info, trying to connect to:", sensorRight
  r = Peripheral(sensorRight)


except BTLEException:
  print "Fatal, unable to connect!"
  
except:
  print "Fatal, unexpected error!"
  traceback.print_exc()
  raise

else:

  try:
    print "Info, connected and turning sensor on!"
    lh = l.getCharacteristics(uuid=config_uuid)[0]
    lh.write(sensorOn, withResponse=True)

    rh = r.getCharacteristics(uuid=config_uuid)[0]
    rh.write(sensorOn, withResponse=True)
    
    print "Info, reading values!"
    lh = l.getCharacteristics(uuid=data_uuid)[0]
    rh = r.getCharacteristics(uuid=data_uuid)[0]

    t_end = time.time() + time_read

    file_accelerometer = open("quite_accelerometer_pf.csv","w")

    while time.time() < t_end:
      rawValsL = lh.read()
      rawValsR = rh.read()

      
      #for rawVal in rawValsL:
        #temp = ord(rawVal)
        #print "%2.2x" % temp,
    

      # Movement data: 9 bytes made up of x, y and z for Gyro, Accelerometer, 
      # and Magnetometer.  Raw values must be divided by scale
      (gyroX, gyroY, gyroZ, accX, accY, accZ, magX, magY, magZ) = struct.unpack('<hhhhhhhhh', rawValsL)
      (gyroXR, gyroYR, gyroZR, accXR, accYR, accZR, magXR, magYR, magZR) = struct.unpack('<hhhhhhhhh', rawValsR)


      #scale = 128.0
      #print "Gyro - x: %2.2f, y: %2.2f, z: %2.2f" % (gyroX / scale, gyroY / scale, gyroZ / scale)
      
      scale = 4096.0
      file_accelerometer.write("%2.2f,%2.2f,%2.2f,%2.2f,%2.2f,%2.2f,%s\n" % (accX / scale, accY / scale, accZ / scale, \
        accXR / scale, accYR / scale, accZR / scale,label))

      #scale = (32768.0 / 4912.0)
      #print "Mag - x: %2.2f, y: %2.2f, z: %2.2f" % (magX / scale, magY / scale, magZ / scale)
    
    print "Info, turning sensor off!"
    lh = l.getCharacteristics(uuid=config_uuid)[0]
    lh.write(sensorOff, withResponse=True)
    rh = r.getCharacteristics(uuid=config_uuid)[0]
    rh.write(sensorOff, withResponse=True)
    
  except:
    print "Fatal, unexpected error!"
    traceback.print_exc()
    raise

  finally:
    print "Info, disconnecting!"
    l.disconnect()
    r.disconnect()
    
finally:
  quit()