#
# TI SimpleLink SensorTag 2015
# Date: 2015 07 07
#
# Sensor: Gyro, Accelerometer and Magnetometer
# Values: 9 bytes x, y, z for each sensor
# Note: Sensor values have not been validated
#
import struct, sys, traceback, time, threading
from bluepy.btle import UUID, Peripheral, BTLEException
import paho.mqtt.client as mqtt

#broker_address = "test.mosquitto.org"
broker_address = "iot.eclipse.org"
broker_portno = 1883
client = mqtt.Client()

def TI_UUID(val):
    return UUID("%08X-0451-4000-b000-000000000000" % (0xF0000000+val))

config_uuid = TI_UUID(0xAA82)
data_uuid = TI_UUID(0xAA81)

dataL = {}
dataR = {}
dataC = {}

# Bit settings to turn on individual movement sensors
# bits 0 - 2: Gyro x, y z
# bits 3 - 5: Accelerometer x, y, z
# bit: 6: Magnetometer turns on X, y , z with one bit

#gyroOn = 0x0700
#accOn = 0x3800
#magOn = 0xC001

#sensorOnVal = gyroOn | magOn | accOn
sensorOnVal = 0x7F02

sensorOn = struct.pack("BB", 0x7F, 0x02)
sensorOff = struct.pack("BB", 0x00, 0x00)

if len(sys.argv) != 2:
  print("Fatal, must pass label and time in seconds: <time> ")
  quit()

time_read = float(sys.argv[1])

sensors_connected = 0
e = threading.Event()

dicData = {}
tLock = threading.Lock()

def read_data(sensorName,sensorMAC):
    global sensors_connected
    try:
        print ("Info, trying to connect to %s: %s" % (sensorName,sensorMAC))
        sensor = Peripheral(sensorMAC)

    except BTLEException:
        print ("Fatal, unable to connect!")
  
    except:
        print ("Fatal, unexpected error!")
        traceback.print_exc()
        raise
    else:
        try:
            print ("Info, connected and turning sensor %s on!" % sensorName)
            sh = sensor.getCharacteristics(uuid=config_uuid)[0]
            sh.write(sensorOn, withResponse=True) 

            

            with tLock:
                sensors_connected += 1
            e.set()

            while sensors_connected < 3:    
                e.wait()

                

            print ("Info, reading values on %s!" % sensorName)

            sh = sensor.getCharacteristics(uuid=data_uuid)[0]

            t_end = time.time() + time_read

            index = 0
            beginning = time.time()

            

            while time.time() <= t_end:
                time_get_data = time.time() + 0.5
                data = '['
                while time.time() < time_get_data:
                    index = index + 1
                    rawVals = sh.read()

                    #   Movement data: 9 bytes made up of x, y and z for Gyro, Accelerometer, 
                    #   and Magnetometer.  Raw values must be divided by scale
                    (gyroX, gyroY, gyroZ, accX, accY, accZ, magX, magY, magZ) = struct.unpack('<hhhhhhhhh', rawVals)
                    scale = 4096.0
                    
                    with tLock:
                        if sensorName not in dicData:
                            dicData[sensorName] = {}
                            dicData[sensorName][index] = [accX,accY,accZ]
                        else:
                            dicData[sensorName][index] = [accX,accY,accZ]

                    data += '{\"index\": %d, \"x\": %f, \"y\": %f, \"z\": %f, \"sensor\": \"%s\"},' % (index, accX, accY, accZ, sensorName)
                
                data = data[:-1]
                data += ']'
                client.publish(topic = sensorName, payload = data)
            
            print ("Info, turning sensor %s off!" % sensorName)
            sh = sensor.getCharacteristics(uuid=config_uuid)[0]
            sh.write(sensorOff, withResponse=True)
            sensors_connected -= 1
        except:
            print ("Fatal, unexpected error!")
            traceback.print_exc()
            raise

        finally:
            print ("Info, disconnecting!")
            sensor.disconnect()
    finally:
        quit()


    

sensorLeft = '24:71:89:BB:FA:00'
sensorRight = '24:71:89:C0:BC:84'
sensorChest = '24:71:89:C1:3C:04'

client.connect(broker_address, broker_portno)

lThread = threading.Thread(target=read_data,args=("Left",sensorLeft,))
lThread.start()
rThread = threading.Thread(target=read_data,args=("Right",sensorRight,))
rThread.start()
cThread = threading.Thread(target=read_data,args=("Chest",sensorChest,))
cThread.start()

lThread.join()
print("End of left thread!!")
rThread.join()
print("End of right thread!!")
cThread.join()
print("End of chest thread!!")