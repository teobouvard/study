const int ledPin = 13;      // LED connected to digital pin 13
const int knockSensor0 = A2; // the piezo is connected to analog pin 2
const int threshold = 200;  // threshold value to decide when the detected sound is a knock or not

String dataSent;
String num0 = "0";
float accumulateur = 0;
int delta = 0;

int sensorReading0 = 0;// variable to store the value read from the sensor pin
int last_sensorReading0 = 0;
int ledState = LOW;          // variable used to store the last LED status, to toggle the light

void setup() {
  //pinMode(ledPin, OUTPUT); // declare the ledPin as as OUTPUT
  Serial.begin(9600);       // use the serial port
}

void loop() {

  sensorReading0 = analogRead(knockSensor0);
  delta = sensorReading0 - last_sensorReading0;

  if(delta > 0){
    accumulateur += delta;
  }
  else{
    if(accumulateur > threshold){
      dataSent = num0 + round(accumulateur/1024.0*9.0);
      Serial.print(num0);
    }
    accumulateur = 0;
  }

  last_sensorReading0 = sensorReading0;
}
