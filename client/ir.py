"""Demo script to read values from an IR motion sensor."""

import RPi.GPIO as GPIO
import time

# Pin Definitions
input_pin = 18  # BCM pin 18, BOARD pin 12


def main():
    """Poll the sensor and print status changes."""
    prev_value = None

    # Pin Setup:
    GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
    GPIO.setup(input_pin, GPIO.IN)  # set pin as an input pin
    print("Starting demo now! Press CTRL+C to exit")
    try:
        while True:
            value = GPIO.input(input_pin)
            if value != prev_value:
                if value == GPIO.LOW:
                    value_str = "Tracking human..."
                else:
                    value_str = "Human exited"
                print(value_str)
                prev_value = value
            time.sleep(1)
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
