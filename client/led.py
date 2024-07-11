import Jetson.GPIO as GPIO
import time

# Pin Definitions
led_pin = 18  # BOARD pin 12, BCM pin 18

# Set up the GPIO channel
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)

# Blink the LED
try:
    while True:
        GPIO.output(led_pin, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(led_pin, GPIO.LOW)
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
