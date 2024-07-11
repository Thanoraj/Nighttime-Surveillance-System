import Jetson.GPIO as GPIO
import time

# Pin Definitions
ir_sensor_pin = 17  # Pin 11 on the header, GPIO17

# Pin Setup
GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
GPIO.setup(ir_sensor_pin, GPIO.IN)  # IR sensor pin set as input

try:
    print("Starting IR sensor monitoring...")
    while True:
        if GPIO.input(ir_sensor_pin):
            print("Object detected!")
        else:
            print("No object detected.")
        time.sleep(0.5)  # Delay for half a second
except KeyboardInterrupt:
    print("Exiting gracefully.")
finally:
    GPIO.cleanup()  # Cleanup all GPIO
