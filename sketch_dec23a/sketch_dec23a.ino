#include <DHT.h>

// Define sensor pins
#define DHTPIN 2          // DHT11 data pin connected to digital pin 2
#define SOIL_PIN A0       // Soil moisture sensor analog output to A0

// Define sensor type
#define DHTTYPE DHT11     

// Initialize DHT sensor
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  dht.begin();
  Serial.println("DHT11 and Soil Moisture Sensor Live Data");
}

void loop() {
  // Read humidity and temperature
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature(); // Celsius

  // Read soil moisture
  int soilValue = analogRead(SOIL_PIN);
  int soilPercent = map(soilValue, 1023, 0, 0, 100);

  // Check if any reads failed
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Error reading DHT sensor!");
    return;
  }

  // Output in CSV format: temperature,humidity,moisture
  Serial.print(temperature);
  Serial.print(",");
  Serial.print(humidity);
  Serial.print(",");
  Serial.println(soilPercent);

  delay(2000);
}
